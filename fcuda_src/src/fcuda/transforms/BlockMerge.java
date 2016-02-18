package fcuda.transforms;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import cetus.analysis.DataFlow;
import cetus.exec.Driver;
import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FlatIterator;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbolic;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.hir.WhileLoop;
import fcuda.utils.*;
import fcuda.transforms.UnrollThreadLoops.StatementNames;
import fcuda.ir.ThreadLoop;
import fcuda.common.*;
import fcuda.transforms.KernelTransformPass;

public class BlockMerge extends KernelTransformPass {

  public boolean bmerge; // boolean variable that indicates merge or not.

  // # of parallel thread blocks after merging,
  // parameter for duplicate phase.
  public int parallelTb; 

  public int numTbMerge; // # of thread blocks to merge.

  public LinkedList<String> dupVars; // the list to be duplicated (not shared)

  public LinkedList<String> sharedVars; // the list of shared arrays

  private List<String> mPartitionArrays;

  // new generated variables that need to be
  // duplicated.
  private Set<Expression> newDef; 


  private LinkedList<VariableDeclaration> allDecl;

  public CompoundStatement newProcBody;

  public LinkedList<String> compute_tasks;

  public LinkedList<String> allKernelCall;

  public LinkedList<ExpressionStatement> oldKernelCall;

  public int stage;

  private int mempartFactor;

  public static String Block_Suffix = "_m";

  public int totalBram;   // total bram on the chip.

  public int coreBram;    // the required BRAM for each core before thread block merging.

  public int sharedBram;  // the required BRAM for the shared data. 

  public int totalBlock;  // total number of thread blocks.

  public enum StatementNames {
    DefaultStatement, ExpressionStatement, WhileLoop, DeclarationStatement, CompoundStatement, AnnotationStatement, ForLoop, ThreadLoopStatement, IfStatement
  }

  private StatementNames getStmtTypeId(Statement stmt) {
    if (stmt instanceof WhileLoop)
      return StatementNames.WhileLoop;
    if (stmt instanceof ExpressionStatement)
      return StatementNames.ExpressionStatement;
    if (stmt instanceof CompoundStatement)
      return StatementNames.CompoundStatement;
    if (stmt instanceof DeclarationStatement)
      return StatementNames.DeclarationStatement;
    if (stmt instanceof AnnotationStatement)
      return StatementNames.AnnotationStatement;
    if (stmt instanceof ForLoop)
      return StatementNames.ForLoop;
    if (stmt instanceof ThreadLoop)
      return StatementNames.ThreadLoopStatement;
    if (stmt instanceof IfStatement)
      return StatementNames.IfStatement;
    return StatementNames.DefaultStatement;
  }

  // collect the data required to optProb
  /*
   * 1. which data should be duplicated, then the others should be shared data
   * (including constant as well)
   */
  public void init(Procedure proc) {

    dupVars = null;
    dupVars = new LinkedList<String>();

    // 1. Matrix Multiplication. 
    List<String> temp = FCUDAutils.getTaskNonSharedArray(proc);
    if(temp != null)
    {
      System.out.println("dupVars:" + temp);
      dupVars.addAll(temp);
    }

    sharedVars = null;
    sharedVars = new LinkedList<String>();
    temp = FCUDAutils.getTaskSharedArray(proc);
    if(temp != null)
    {
      System.out.println("sharedVars:" + temp);
      sharedVars.addAll(temp);
    }

    //dupVars.add("Bs");
    //dupVars.add("Csub_block");

    // 2. CP.
    //dupVars.add("energyval_block");

    // always included for all the benchmarks.
    dupVars.add("blockIdx");

    newProcBody = null;
    newProcBody = new CompoundStatement();

    newDef = null;
    newDef = new HashSet();
  }

  /*   
   * Assumption: number of BRAM is the limiting factor, limit the parallelism.
   * 
   * In FCUDA2: we have already chosen a good unroll, mpart and core number; these parameters exhibit a good balance between fine grained parallelism (unroll, mpart) 
   * and coarse grained parallelism (core).  
   * 
   * In theory, once we choose a merging number (numTbMerge), we have to invoke fcuda2 again. but this is very expensive and may not be necessary. 
   * Because the way we merge thread blocks does not affect the clock cycles, thus the unroll number and mpart number should still be good. 
   * Thus, we can keep the unroll and mpart number solution from FCUDA2 (memory bandwidth is sufficient for computation). 
   * 
   * So, we just need to update the core number.  
   * 
   * Inputs:  
   * 1. totalBram:  number of available brams on the FPGA.
   * 2. coreBram:   the required BRAM for each core before thread block merging. To be accurate, we should use the ISE data.
   * 3. sharedBram: the required BRAM for the shared data. To be accurate, we should use the ISE data.
   4. totalBlock: number of thread blocks.

Outputs:
1.parallelTb:     number of parallel thread blocks (core number) after thread block merging.
2.numTbMerge(N):  number of thread blocks (original) to be merged.

parallelTb x numTbMerge = the total number of thread blocks (original) running in parallel.

should consider the totalbram, the round of executions.

   * Optimization objective 
   * 
   * B = N x coreBram - (N - 1) x sharedBram;  B is the bram usage after merging N blocks together. 
   * C = N x floor(totalBram/br);  C is the concurrency, the number of thread blocks (original) running in parallel.
   * T = ceil(totalBlock/C);  

   * Examples: totalBram (10), coreBram(2), sharedBram(1).
   * N = 1, parallelism = 5;
   * N = 2, parallelism = 6;
   * N = 3, parallelism = 6;
   * N = 4, parallelism = 8;
   * N = 5, parallelism = 5;
   */ 

  public void optProb(Procedure proc) {

    //1. MatrixMul: solution: unroll: 4 mpart:2.
    //totalBlock = 4096; coreBram   = 10; sharedBram = 2;
    //2. CP: solution: unroll: 4 mpart:2.
    //totalBlock = 1024; coreBram = 24; sharedBram = 20;

    totalBlock = FCUDAutils.getTaskTotalBlock(proc);
    totalBram  = FCUDAutils.getTaskTotalBram(proc);
    coreBram   = FCUDAutils.getTaskCoreBram(proc);
    sharedBram = FCUDAutils.getTaskSharedBram(proc);

    System.out.println("optProb:" + totalBram + " " + coreBram + " " + sharedBram + " " + totalBlock);

    int bram, concur, round;
    int bestRound = totalBlock;
    for(int n = 1; n <= totalBlock; n++)
    {
      bram    = n * coreBram - (n - 1) * sharedBram;
      if(bram > totalBram)
        break;
      concur  = n * (totalBram/bram); 
      round   = totalBlock / concur;
      if(totalBlock % concur != 0)
        round++;
      System.out.println("n: " + n + " bram:" + bram + " concurrency: " + concur + " round: " + round);
      if(round < bestRound)
      {
        bestRound  = round;
        numTbMerge = n;
        parallelTb = totalBram/bram;
      }
    }

    System.out.println("BlockMerge Solution");
    System.out.println("# of thread blocks to merge " + numTbMerge);
    System.out.println("# of cores in parallel " +  parallelTb);
  }

  public BlockMerge(Program program) {
    super(program);

    stage = 0;
    bmerge = true;	
  }

  public boolean inArrayPartList(String name) {

    if(mPartitionArrays == null)
      return false;

    Iterator iter = mPartitionArrays.iterator();
    while (iter.hasNext()) {
      String str = (String) iter.next();
      if (str.equals(name))
        return true;
    }
    return false;
  }

  public int getCallIndex(String name){
    int index = 0;
    Iterator iter = allKernelCall.iterator();
    while (iter.hasNext()) {
      String str = (String) iter.next();
      if(str.compareTo(name) == 0)
        index++;
    }
    return index;
  }

  public boolean isComputeCall(String name) {
    Iterator iter = compute_tasks.iterator();
    while (iter.hasNext()) {
      String str = (String) iter.next();
      if (name.compareTo(str) == 0)
        return true;
    }
    return false;
  }

  /*
   * Note: suffix might be added to dupVars to form name.
   */
  public boolean inDupList(String name) {
    Iterator iter = dupVars.iterator();
    while (iter.hasNext()) {
      String str = (String) iter.next();
      if (name.startsWith(str))
        return true;
    }
    return false;
  }

  public String inSharedList(String name) {
    Iterator iter = sharedVars.iterator();
    while (iter.hasNext()) {
      String str = (String) iter.next();
      if (name.startsWith(str))
        return str;
    }
    return null;
  }

  public boolean isObjectAffected(Object obj) {

    Iterator it = dupVars.iterator();
    while (it.hasNext()) {
      String str = (String) it.next();
      if (obj.toString().startsWith(str))
        return true;
    }

    if (newDef == null)
      return false;

    it = newDef.iterator();
    while (it.hasNext()) {
      Expression expr = (Expression) it.next();
      if (expr.equals(obj))
        return true;
    }
    return false;
  }

  public boolean isStatementAffected(Statement cStmt) {
    boolean bcopy = false;
    DepthFirstIterator iter = new DepthFirstIterator(cStmt);
    while (iter.hasNext()) {
      Object child = iter.next();
      if (isObjectAffected(child)) {
        bcopy = true;
        break;
      }
    }
    return bcopy;
  }

  public boolean isExpressionAffected(Expression expr) {
    boolean bcopy = false;
    DepthFirstIterator iter = new DepthFirstIterator(expr);
    while (iter.hasNext()) {
      Object child = iter.next();
      if (isObjectAffected(child)) {
        bcopy = true;
        break;
      }
    }
    return bcopy;
  }

  public Expression createSuffixExpr(Expression index, int factor) {

    if (index instanceof IDExpression) {
      IDExpression ide = (IDExpression) index;
      String name = ide.toString() + Block_Suffix + factor;
      NameID id = new NameID(name);
      return id;
    } else
      return index;
  }

  public Expression createBlockExpr(Expression index, int factor) {

    if (index instanceof IDExpression) {
      IDExpression ide = (IDExpression) index;
      String name = ide.toString();
      int pos = name.lastIndexOf("_block") + 6;
      name = name.substring(0, pos);
      name = name + factor;
      NameID id = new NameID(name);
      return id;
    } else
      return index;
  }

  public void collectAllDeclaration(Procedure proc) {
    allDecl = new LinkedList();
    DepthFirstIterator diter = new DepthFirstIterator(proc.getBody());
    diter.pruneOn(Expression.class);
    while (diter.hasNext()) {
      Object child = diter.next();

      if (child instanceof VariableDeclaration)
        allDecl.add((VariableDeclaration) child);
    }

    System.out.println("[All Declaration]" + allDecl.toString());
  }

  public List<Specifier> getVariableType(IDExpression ide) {
    Iterator iter = allDecl.iterator();
    while (iter.hasNext()) {
      VariableDeclaration decl = (VariableDeclaration) iter.next();
      List<IDExpression> list = decl.getDeclaredIDs();

      Iterator it = list.iterator();
      while (it.hasNext()) {
        IDExpression ide0 = (IDExpression) it.next();
        if (ide0.toString().equals(ide.toString())) {

          return decl.getSpecifiers();
        }
      }
    }


    List<Specifier> l = new ArrayList<Specifier>();
    l.add(Specifier.INT);
    return l;
  }

  public void insertVarDecl(CompoundStatement procbody) {

    // the newly generated variables must be in newDef, not in DupVar.
    for (Expression expr : newDef) {
      if (expr instanceof ArrayAccess)
        continue;

      if (expr instanceof IDExpression) {
        System.out.println("[var]" + expr.toString());

        for (int i = 0; i < numTbMerge; i++) {
          Expression newID = createSuffixExpr(expr, i);
          Declarator declor = new VariableDeclarator(
              (IDExpression) newID);
          Declaration decl = new VariableDeclaration(
              getVariableType((IDExpression) expr), declor);

          procbody.addDeclaration(decl);
        }
      }
    }
  }

  private void expandDeclarationStatement(DeclarationStatement stmt,
      CompoundStatement newBody) {

    if (stage == 1) {
      if (newBody != null)
        newBody.addANSIDeclaration(stmt.getDeclaration().clone());
    } else {

    }
  }

  private void expandDefaultStatement(Statement stmt,
      CompoundStatement newBody) {

    if (stage == 1) {
      if (isStatementAffected(stmt)) {
        // we may define some new variables, so if it is true, update!!!
        Set<Expression> defSet = DataFlow.mayDefine(stmt);
        Iterator it = defSet.iterator();
        while (it.hasNext()) {
          Expression expr = (Expression) it.next();

          if (expr instanceof ArrayAccess)
            continue;

          newDef.add(expr);
        }

        for (int i = 0; i < numTbMerge; i++) {
          Statement newS = (Statement) stmt.clone();
          List old_Exprs = new ArrayList();
          DepthFirstIterator newIter = new DepthFirstIterator(newS);
          while (newIter.hasNext()) {
            Object child = newIter.next();
            if (child instanceof Expression
                && isObjectAffected(child)) {
              old_Exprs.add(child);
                }
          }

          for (Object old_expr : old_Exprs) {
            Expression new_expr = null;
            new_expr = createSuffixExpr((Expression) old_expr, i);
            ((Expression) old_expr).swapWith(new_expr);

          }
          newBody.addStatement(newS);
        }

      } else
        newBody.addStatement(stmt.clone());
    } else {

    }
  }

  private void expandForloopStatement(ForLoop stmt, CompoundStatement newBody) {

    Statement init = stmt.getInitialStatement();
    Expression cond = stmt.getCondition();
    Expression step = stmt.getStep();
    CompoundStatement loopbody = new CompoundStatement();
    ForLoop forloop = null;

    if (stage == 1) {
      if ((isStatementAffected(init) || isExpressionAffected(cond) || isExpressionAffected(step))) {

        expandDefaultStatement(stmt, newBody);

      } else {

        forloop = new ForLoop(init.clone(), cond.clone(), step.clone(),
            loopbody);

        newBody.addStatement(forloop);

        expandCompoundStatement((CompoundStatement) stmt.getBody(),
            loopbody);
      }
    } else {

      expandCompoundStatement((CompoundStatement) stmt.getBody(), newBody);
    }
  }

  private void expandThreadLoopStatement(Statement cStmt) {

    if (stage == 1) {
      ThreadLoop tloop = (ThreadLoop) cStmt;

      expandCompoundStatement((CompoundStatement) tloop.getBody(),
          newProcBody);

      // update variable declaration.
      insertVarDecl(newProcBody);

      tloop.setBody(newProcBody);
    } else {

    }
  }

  private void expandIfStatement(IfStatement stmt, CompoundStatement newBody) {

    Boolean belse = false;
    Expression control = stmt.getControlExpression();
    CompoundStatement thenStmt = new CompoundStatement();
    CompoundStatement elseStmt = null;

    if (stmt.getElseStatement() != null)
      belse = true;

    if (belse)
      elseStmt = new CompoundStatement();

    if (stage == 1) {
      if (isExpressionAffected(control)) {

      } else {
        if (belse) {
          newBody.addStatement(new IfStatement(control.clone(),
                thenStmt, elseStmt));
          expandCompoundStatement(
              (CompoundStatement) stmt.getThenStatement(),
              thenStmt);
          expandCompoundStatement(
              (CompoundStatement) stmt.getElseStatement(),
              elseStmt);
        } else {
          newBody.addStatement(new IfStatement(control.clone(),
                thenStmt));
          expandCompoundStatement(
              (CompoundStatement) stmt.getThenStatement(),
              thenStmt);
        }
      }
    } else {

      if (belse) {
        expandCompoundStatement(
            (CompoundStatement) stmt.getThenStatement(), thenStmt);
        expandCompoundStatement(
            (CompoundStatement) stmt.getElseStatement(), elseStmt);
      } else {

        expandCompoundStatement(
            (CompoundStatement) stmt.getThenStatement(), thenStmt);
      }
    }
  }

  private void expandWhileLoopStatement(WhileLoop stmt,
      CompoundStatement newBody) {
    expandCompoundStatement((CompoundStatement) stmt.getBody(), newBody);
  }

  private void handleKernelCall(ExpressionStatement expStmt, CompoundStatement cStmt) {

    FunctionCall coreCall = (FunctionCall) (expStmt.getExpression());
    Expression   exprName = coreCall.getName();
    String       sName    = exprName.toString(); 

    System.out.println("Kernel Call :" + sName);

    if (isComputeCall(sName)) {
      System.out.println("Compute task :" + sName);

      allKernelCall.add(sName);
      int index = getCallIndex(sName);
      if(index <= parallelTb)
      {
        List<Expression> newlist = new LinkedList<Expression>();
        List<Expression> oldlist = coreCall.getArguments();

        Iterator it = oldlist.iterator();
        while (it.hasNext()) {
          Expression expr = (Expression) it.next();

          if(inDupList(expr.toString()))
          {
            for(int i = 0; i < numTbMerge; i++)
            {
              Expression newexpr = expr.clone();
              newexpr = createBlockExpr(newexpr, (index - 1) * numTbMerge + i);
              newlist.add(newexpr);
            }
          }else
            newlist.add(expr.clone());
        }
        FunctionCall newCall  = coreCall.clone();
        newCall.setArguments(newlist);
        coreCall.swapWith(newCall);
      }else{
        oldKernelCall.add(expStmt);
      }
    }else{
      // If it is not compute task, then it must be the fetch wrapper.
      List<Expression> newlist = new LinkedList<Expression>();
      List<Expression> oldlist = coreCall.getArguments();

      System.out.println("mempartFactor:" + mempartFactor + " " + mPartitionArrays);
      Iterator it = oldlist.iterator();
      int cnt = 0;
      while(it.hasNext()){
        Expression expr   = (Expression) it.next();
        String arrayName  = inSharedList(expr.toString());
        if(arrayName != null)
        { 
          int index = 0; 
          if(inArrayPartList(arrayName))
            index = cnt / (numTbMerge * mempartFactor);
          else
            index = cnt / numTbMerge;

          Expression newexpr = expr.clone();
          newexpr = createBlockExpr(newexpr, index);
          newlist.add(newexpr);
          cnt++;
        }else
          newlist.add(expr.clone());
      }
      FunctionCall newCall = coreCall.clone();
      newCall.setArguments(newlist);
      coreCall.swapWith(newCall);
    }

    System.out.println("Kernel Call :" + sName + " Done");
  }

  private void expandCompoundStatement(CompoundStatement cStmt,
      CompoundStatement newBody) {
    if (cStmt == null)
      return;

    FlatIterator flatIter = new FlatIterator(cStmt);

    while (flatIter.hasNext()) {
      Object currObj = flatIter.next();
      if (!(currObj instanceof Statement))
        Tools.exit("Child " + currObj.toString()
            + " of compound statement is not statement");
      Statement currStmt = (Statement) currObj;

      switch (getStmtTypeId(currStmt)) {
        case DeclarationStatement:
          expandDeclarationStatement((DeclarationStatement) currStmt,
              newBody);
          break;
        case ThreadLoopStatement:
          expandThreadLoopStatement(currStmt);
          break;
        case ForLoop:
          expandForloopStatement((ForLoop) currStmt, newBody);
          break;
        case IfStatement:
          expandIfStatement((IfStatement) currStmt, newBody);
          break;
        case CompoundStatement:
          expandCompoundStatement((CompoundStatement) currStmt, newBody);
          break;
        case WhileLoop:
          expandWhileLoopStatement((WhileLoop) currStmt, newBody);
          break;
        case ExpressionStatement:
          if (stage == 2) {
            ExpressionStatement exps = (ExpressionStatement) currStmt;
            Expression exp = exps.getExpression();
            if (exp instanceof FunctionCall) {
              handleKernelCall(exps, cStmt);
            }

          } else
            expandDefaultStatement(currStmt, newBody);
          break;
        default:
          expandDefaultStatement(currStmt, newBody);
          break;

      }
    }
  }

  public void updateProcParameter(Procedure proc) {

    List<Declaration> oldDeclLst = new LinkedList<Declaration>();

    for (Object declObj : proc.getParameters()) {
      if (!(declObj instanceof VariableDeclaration))
        Tools.exit("Parameter:" + declObj.toString()
            + " of Procedure: " + proc.getName().toString()
            + "is not a VariableDeclaration");

      VariableDeclaration pDecl = (VariableDeclaration) declObj;
      Declarator pDeclor = pDecl.getDeclarator(0);
      IDExpression ide = pDeclor.getID();

      if (inDupList(ide.toString())) {

        LinkedList<Specifier> leadSpecs = new LinkedList<Specifier>();
        LinkedList<Specifier> trailSpecs = new LinkedList<Specifier>();
        LinkedList<Specifier> leadSpecsDecl = new LinkedList<Specifier>();

        trailSpecs.addAll(pDeclor.getArraySpecifiers());
        leadSpecs.addAll(pDeclor.getSpecifiers());
        leadSpecsDecl.addAll(((VariableDeclaration) pDecl)
            .getSpecifiers());

        for (int i = numTbMerge - 1; i >= 0; i--) {

          IDExpression newPartId = new NameID(pDeclor.getID()
              .toString() + Block_Suffix + i);
          VariableDeclarator newPartition = new VariableDeclarator(
              leadSpecs, newPartId, trailSpecs);
          VariableDeclaration newdecl = new VariableDeclaration(
              leadSpecsDecl, newPartition);
          proc.addDeclarationAfter(pDecl, (Declaration) newdecl);
        }

        oldDeclLst.add(pDecl);
      }
    }

    for (Declaration oldDecl : oldDeclLst)
      proc.removeDeclaration(oldDecl);

  }

  public void updateProcBody(Procedure proc) {

    expandCompoundStatement(proc.getBody(), null);
  }

  public void merge_task(Procedure proc) {

    System.out.println("merge_task:" + proc.getName().toString());

    init(proc);
    collectAllDeclaration(proc);

    // 1. update procedure parameter.
    updateProcParameter(proc);
    // 2. update the procedure body.
    updateProcBody(proc);
  }

  public void merge_top(Procedure proc) {
    /*
     * So far, the total number of threads blocks (numTbMerge x parallelTb)
     * has been duplicated by the duplication pass. We need to do the
     * following in this pass. 1. update the procedure call site (only use
     * one copy of shared variables). 2. update the fetch/write function. 3.
     * Based on experiments, we find that Autopilot will automatically
     * remove the variables which are not used. so, we do not need to bother
     * to remove the duplicate copy for the shared variables.
     */
    System.out.println("merge_top:" + proc.getName().toString());

    expandCompoundStatement(proc.getBody(), null);

    Iterator it = oldKernelCall.iterator();
    while (it.hasNext()) {
      Statement stmt = ((Statement) it.next());
      stmt.detach();
    }

  }


  public void  transformProcedure(Procedure proc) {

    System.out.println(proc.getName().toString() + " Stage:" + stage);

    if (stage == 0) {

      List<Procedure> tskLst = FCUDAutils.getTaskMapping(proc
          .getSymbolName());

      if (tskLst != null) {
        for (Procedure task : tskLst) {
          if (FCUDAutils.getTaskType(task).equals("compute")) {

            /*
             *  solve the optimization problem, the bram and core parameters are associated with the compute task using pragmas.
             */
            if(FCUDAutils.getTaskSharedArray(task) == null)
            {
              System.out.println("not merge");
              bmerge = false;
              return;
            }

            optProb(task); // solve the optimization problem.
            break;
          }
        }
      }

      int num = parallelTb * numTbMerge;
      // if there are shared data 
      FCUDAGlobalData.setNumParallelThreadBlocks(proc, num);

    } else {

      compute_tasks = null;
      allKernelCall = null;
      oldKernelCall = null;
      compute_tasks = new LinkedList<String>();
      allKernelCall = new LinkedList<String>();
      oldKernelCall = new LinkedList<ExpressionStatement>();

      List<Procedure> tskLst = FCUDAutils.getTaskMapping(proc
          .getSymbolName());

      if (tskLst != null) {
        for (Procedure task : tskLst) {
          if (FCUDAutils.getTaskType(task).equals("compute")) {
            System.out.println("Task:"  +task.getName().toString());

            mempartFactor = FCUDAutils.getTaskMempart(task);
            mPartitionArrays = FCUDAutils.getTaskSplitArray(task);
            System.out.println("mempartFactor: " + mempartFactor + " " + mPartitionArrays);

            compute_tasks.add(task.getName().toString());
            merge_task(task);
          }else{

            System.out.println("transfer task" + task.getName().toString());
          }
        }
      }

      stage++;
      merge_top(proc);

    }

  }

  public String getPassName() {
    return new String("[BLOCKMERGE---MCUDA]");
  }

  public void updateStage(){
    stage++;
  }
}
