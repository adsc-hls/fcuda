//============================================================================//
//    FCUDA
//    Copyright (c) <2016> 
//    <University of Illinois at Urbana-Champaign>
//    <University of California at Los Angeles> 
//    All rights reserved.
// 
//    Developed by:
// 
//        <ES CAD Group & IMPACT Research Group>
//            <University of Illinois at Urbana-Champaign>
//            <http://dchen.ece.illinois.edu/>
//            <http://impact.crhc.illinois.edu/>
// 
//        <VAST Laboratory>
//            <University of California at Los Angeles>
//            <http://vast.cs.ucla.edu/>
// 
//        <Hardware Research Group>
//            <Advanced Digital Sciences Center>
//            <http://adsc.illinois.edu/>
//============================================================================//

package fcuda.transforms;

import java.util.*;

import fcuda.utils.*;
import fcuda.analysis.*;
import fcuda.ir.*;
import fcuda.common.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;

/*
 * Use the user-inserted FCUDA annotation to split the kernel
 * into computation and communication tasks.
 * Each FCUDA annotated task is converted into an independent
 * procedure that is called from the kernel function.
 * The parameters of each task procedure are determinded through
 * dataflow analysis
 */

public class SplitFcudaTasks extends KernelTransformPass
{

  public class TransferInfo
  {
    private int mNumTransferCores;
    private int mTransferDirection;
    private IDExpression bramId;
    private IDExpression ptrId;
    private Statement transferStmt;

    // Can be variable name/macro
    private String mTransferSize;

    TransferInfo (Statement transStmt, IDExpression brmId, IDExpression pointId, int dir, String size, int cores)
    {
      setAttributes(transStmt, brmId, pointId, dir, size, cores);
    }

    public void setAttributes(Statement transStmt, IDExpression brmId, IDExpression pointId, int dir, String size, int cores)
    {
      mNumTransferCores= cores;
      mTransferDirection=dir;
      mTransferSize=size;
      bramId = brmId;
      ptrId = pointId;
      transferStmt = transStmt;
    }

    String getSize() { return mTransferSize;	}
    int getDir() { return mTransferDirection; }
    int getNumCores() { return mNumTransferCores; }
    IDExpression getBram() { return bramId; }
    IDExpression getPtr() { return ptrId; }
    Statement getStmt() { return transferStmt; }
  }

  private Procedure mProcedure;
  private int numDims;
  private HashSet<TransferInfo> mTransfInfoSet;
  private HashSet<VariableDeclaration> mToMakeVolatile;
  private HashMap<IDExpression, Integer> mBRAMSet;   // BRAM IDExpression --> Dimensionality
  private HashSet<Declaration> mBRAMDecl;

  private void parseBaseTerms(String baseString, Expression[] baseTerms, SymbolTable st)
  {
    int i;
    String[] result	= baseString.split("\\/");
    for (i=0;i<result.length;++i) {
      if(result.equals(""))
        baseTerms[i] = null;
      else {
        IDExpression idExpr = new NameID(result[i]);
        VariableDeclaration varDecl = FCUDAutils.getVariableDeclarationUnchecked(idExpr, st);
        if (varDecl == null)
          Tools.exit("Symbol "+idExpr.toString()+" is not a variable in transfer pragma - what is it?");
        assert (varDecl.getNumDeclarators() == 1);
        baseTerms[i] = new Identifier((Symbol)varDecl.getDeclarator(0));
      }
    }
  }


  public String getPassName()
  {
    return new String("[SplitFcudaTasks-FCUDA]");
  }

  public SplitFcudaTasks(Program program)
  {
    super(program);
    mTransfInfoSet = new HashSet();
    mToMakeVolatile = new HashSet();
    mToMakeVolatile.clear();
    mBRAMSet = new HashMap();
    mBRAMSet.clear();
    mBRAMDecl = new HashSet();
    mBRAMDecl.clear();
  }

  // Find definition of expression if it is based on ThreadIdx variables. Else return null
  private Statement getVarDef(Procedure proc, Statement tCall, Expression id)
  {
    Statement retStmt = null;
    DepthFirstIterator iter = new DepthFirstIterator(proc);
    iter.pruneOn(Expression.class);

    while(iter.hasNext()) {
      Statement stmt = null;

      try {
        stmt = (Statement)iter.next(Statement.class);
      }
      catch (NoSuchElementException e) {
        break;
      }

      // Stop checking for definitions at the task call statement
      if (stmt.equals(tCall))
        break;

      // Looking only for assignments
      if (!(stmt instanceof ExpressionStatement))
        continue;

      Expression expr = ((ExpressionStatement)stmt).getExpression();
      if (!(expr instanceof AssignmentExpression))
        continue;

      Set<Expression> defSet = DataFlowTools.getDefSet(expr);
      Set<Expression> useSet = DataFlowTools.getUseSet(expr);

      if (defSet.contains(id))
      {
        List<Expression> threadIds = MCUDAUtils.Tidx.getId();
        Iterator <Expression> lItr = threadIds.iterator();
        while (lItr.hasNext())
        {
          if (useSet.contains(lItr.next()))
          {
            retStmt = stmt;
            break;
          }
        }
      }

    }

    return retStmt;
  }


  /** *KG* - for my understanding
   *  taskDecls - actual declarations in function defn - (int* a)
   *  taskArgs - args to be used in function call - taskArgs
   *  defStmts - statement depending on threadIdx that affects the variables being considered here
   */
  // All pointers/arrays are part of arguments - statements that define scalars can be carried over to new function
  private void buildTaskVars(Procedure proc, Set<Expression> useSet, List<IDExpression> taskArgs, Set<IDExpression> taskArgSet,
      List<Declaration> taskDecls, List<Statement> defStmts,
      Statement tCall)
  {
    // Build task parameter list
    for (Expression tmpExpr : useSet) {
      System.out.println("tmpExpr: " + tmpExpr.toString());
      System.out.println("  of class: " + tmpExpr.getClass().toString());

      Expression tmpExpr2 = tmpExpr;

      if (tmpExpr instanceof ArrayAccess) {
        tmpExpr2 = ((ArrayAccess)tmpExpr).getArrayName();
      }

      if (tmpExpr instanceof AccessExpression) {
        tmpExpr2 = ((AccessExpression)tmpExpr).getLHS();
        if (tmpExpr2 instanceof ArrayAccess)
          tmpExpr2 = ((ArrayAccess)tmpExpr2).getArrayName();
        System.out.println("WARNING: currently do not handle RHS elements of AccessExpression "+tmpExpr.toString()+
            " in buildTaskVars()");
      }

      // Ignore CUDA threadIdx variables
      if (FCUDAutils.isCudaDim3Struct(tmpExpr))
        continue;

      // Ignore __syncthreads (other called functions are inlined).
      // The only remaining function calls should be __syncthreads()
      if (tmpExpr instanceof Identifier &&
          tmpExpr.compareTo(MCUDAUtils.getSync().getName()) == 0)
        continue;

      if (!(tmpExpr2 instanceof Identifier)) {
        if (tmpExpr2 instanceof AccessExpression) {
          tmpExpr2 = ((AccessExpression)tmpExpr2).getLHS();
          if (tmpExpr2 instanceof ArrayAccess)
            tmpExpr2 = ((ArrayAccess)tmpExpr2).getArrayName();
        }
        else if (tmpExpr2 instanceof UnaryExpression) {
          tmpExpr2 = ((UnaryExpression)tmpExpr2).getExpression();
        }
        else if (tmpExpr2 instanceof NameID) {
          tmpExpr2 = (IDExpression)new NameID(tmpExpr2.toString());
        }
        else {
          System.out.println(tmpExpr2.toString()+" is not an Identifier ("+tmpExpr2.getClass().toString()+") ");
          System.exit(0);
        }
      }
      IDExpression expr = (IDExpression)tmpExpr2;

      // Find declaration of variable
      // Traversable p = tmpExpr.getParent();
      Traversable p = tCall.getParent();
      Declaration decl=null;
      SymbolTable st = null;
      while (p != null) {
        try {
          st = (SymbolTable)p;
          break;
        } catch (ClassCastException e) {
          p = p.getParent();
        }
      }

      decl = SymbolTools.findSymbol(st, expr);
      if (decl == null) {
        // Check if declaration is contained in task statements (defined in lower level)
        p = tmpExpr.getParent();
        while(p != null) {
          try {
            st = (SymbolTable)p;
            break;
          } catch (ClassCastException e) {
            p = p.getParent();
          }
        }

        decl = SymbolTools.findSymbol(st, expr);
        if (decl != null)
          continue;     // Declaration is contained in task statements (no need to pass as argument)
        else {
          System.out.println("There is no declaration.");
          System.exit(0);
        }
      }

      if (!(decl instanceof VariableDeclaration)) {
        System.out.println("It is not a variable declaration.");
        System.exit(0);
      }
      // Check if it is a global variable
      assert (((VariableDeclaration)decl).getNumDeclarators() == 1);
      try {
        if (SymbolTools.isGlobal((VariableDeclarator)((VariableDeclaration)decl).getDeclarator(0)))
          continue;
      } catch (Exception e) {
        continue;
      }

      if (!(decl instanceof VariableDeclaration)) {
        System.out.println("Undefined declaration.");
        System.exit(0);
      }

      System.out.println("decl: " + decl.toString());


      /** Check if variable is defined as an expression of threadIdx variables
       * FIXME: *AP* This should be made more generic for variables that are indirectly
       *             dependent on threadIdx
       * FIXME: *AP* Does not handle multiple reaching definitions (through loopback paths)
       */

      Statement defStmt;
      defStmt = getVarDef(proc, tCall, expr);
      if (defStmt != null) {
        defStmts.add(defStmt);
        System.out.println("defStmt: " + defStmt.toString());
      }
      else {
        System.out.println("defStmt: none");
        if (!taskArgSet.contains(expr))
        {
          // **AP** Create new declarator without any initializer
          assert (((VariableDeclaration)decl).getNumDeclarators() == 1);
          VariableDeclarator clonedDeclor = (VariableDeclarator)((VariableDeclaration)decl).getDeclarator(0).clone();
          clonedDeclor.setInitializer(null);
          VariableDeclaration declion = new VariableDeclaration(((VariableDeclaration)decl).getSpecifiers(), clonedDeclor);

          // Update lists
          taskDecls.add((Declaration)declion);
          IDExpression cloneExpr = (IDExpression)expr.clone();
          taskArgs.add(cloneExpr);
          taskArgSet.add(expr);
        }
      }
    }

  } // buildTaskVars()



  // **AP** Find BRAM that is associated with off-chip pointer and tranfer direction
  private IDExpression findBRAM(IDExpression ptrId, int dir, Statement transfStmt) {

    IDExpression bramId;
    Set<IDExpression> expSet;
    Expression lftExp, rgtExp;
    Expression ptrExp, bramExp = null;

    // Find assignment representing transfer
    List<AssignmentExpression> assgnExpLst = IRTools.getDescendentsOfType(transfStmt, AssignmentExpression.class);
    System.out.println("[findBRAM]: Annotated Statement --> "+transfStmt.toString());
    System.out.println("---Assignments---  "+assgnExpLst.toString());

    for (AssignmentExpression aExp : assgnExpLst) {
      lftExp = null;
      rgtExp = null;
      ptrExp = null;
      expSet = null;

      if (dir == 1) { // Case of *Write* to off-chip ptrId
        // Identify off chip pointer
        lftExp = aExp.getLHS();
        if (lftExp instanceof ArrayAccess) { // Array
          ptrExp = ((ArrayAccess)lftExp).getArrayName();

          // Check if annotation and statement refer to the same pointer
          if (ptrExp.compareTo((Expression) ptrId) != 0)
            continue;

        } else if (lftExp instanceof UnaryExpression) {  // Pointer
          if (Tools.findExpression(((UnaryExpression)lftExp).getExpression(), (Expression) ptrId) == null)
            continue;
        } else      // Otherwise
          continue;

        // Identify local memory (BRAM)
        rgtExp = aExp.getRHS();
        expSet = mBRAMSet.keySet();
        for (IDExpression brm : expSet) {
          if ((bramExp = Tools.findExpression(rgtExp, (Expression) brm)) != null)
            return (IDExpression)bramExp;
        }
      }

      if (dir == 0) { // Case of *Read* from off-chip ptrId
        // Identify off chip pointer
        rgtExp = aExp.getRHS();
        if (rgtExp instanceof ArrayAccess) { // Array
          ptrExp = ((ArrayAccess)rgtExp).getArrayName();
          // Check if annotation and statement refer to the same pointer
          if (ptrExp.compareTo((Expression) ptrId) != 0)
            continue;
        } else if (rgtExp instanceof UnaryExpression) {  // Pointer
          if (Tools.findExpression(((UnaryExpression)rgtExp).getExpression(), (Expression) ptrId) == null)
            continue;
        } else // Otherwise
          continue;

        // Identify local memory (BRAM)
        lftExp = aExp.getLHS();
        expSet = mBRAMSet.keySet();
        for (IDExpression brm : expSet) {
          if ((bramExp = Tools.findExpression(lftExp, (Expression) brm)) != null)
            return (IDExpression)bramExp;
        }
      }
    }

    return (IDExpression)bramExp;



  } // findBRAM()

  /** *KG* - add parameters to fetch/write functions
   * 	All pointers
   * 	New on-chip arrays based on number of cores
   *  Off-chip access index
   *  Create memcpy call (within threadloop) and instantiate in task procedure
   */
  private void addTransferParameters(Procedure proc, FcudaAnnotation transferAnnot,
      List<Declaration> taskDecls,
      List<IDExpression> taskArgs, Set<IDExpression> taskArgSet,
      CompoundStatement fcTask, LinkedList<Integer> commonArgsIndex )
  {
    int i = 0;
    LinkedList<String> mOffchipPtrNameList = new LinkedList();
    LinkedList<String> coresList = new LinkedList();
    LinkedList<String> sizeList = new LinkedList();
    LinkedList<String> dirList = new LinkedList();
    LinkedList<String> baseList = new LinkedList();
    LinkedList<String> boundList = new LinkedList();
    baseList.clear();

    System.out.println("\n ... Handling transfer params for \n"+transferAnnot.toString());

    FCUDAutils.parseList((String)transferAnnot.get("pointer"), mOffchipPtrNameList);
    FCUDAutils.parseList((String)transferAnnot.get("cores"), coresList);
    FCUDAutils.parseList((String)transferAnnot.get("size"), sizeList);
    FCUDAutils.parseList((String)transferAnnot.get("dir"), dirList);
    String taskName = (String) transferAnnot.get("name");

    if (transferAnnot.get("base") != null)
      FCUDAutils.parseList((String)transferAnnot.get("base"), baseList);

    String ptrName, bramName;
    int dir, cores=1;
    int bound = 0;
    String size;

    Iterator ptrIter = mOffchipPtrNameList.iterator();
    Iterator coresIter = coresList.iterator();
    Iterator sizeIter = sizeList.iterator();
    Iterator dirIter = dirList.iterator();
    Iterator baseIter = baseList.iterator();

    LinkedList<Expression> arrayDims = new LinkedList();
    LinkedList<Specifier> leadSpecList = new LinkedList();
    LinkedList<Specifier> trailSpecList = new LinkedList();

    FunctionCall memcpyCall = null;
    LinkedList<Expression> memcpyArgs = null;

    // Get next executable statement following annotation statement
    // *AP* TRANSFER Annotations should be placed just before the actual transfer statement
    // *AP* Unless it is a transfer related to __constant__ memory or something else (texture memory?)
    // getAnnotatable() returns an empty statement, i.e. not the next statement after the pragma statement
    // In FCUDA Transfer annotations are not attachable, i.e. they do not get attached to the next statement

    Annotatable annotable = transferAnnot.getAnnotatable();
    CompoundStatement cStmt = null;
    Statement annotStmt = null;
    Statement transferStmt = null;
    if (annotable instanceof Statement) {
      annotStmt = (Statement)annotable;
      cStmt = (CompoundStatement)Tools.getAncestorOfType(annotStmt, CompoundStatement.class);
      transferStmt = annotStmt;
    }
    else {
      System.out.println("Could not find annotable statement for "+annotable.getClass().toString());
      System.exit(0);
    }

    while(ptrIter.hasNext()) {
      assert sizeIter.hasNext();
      // **AP** assert coresIter.hasNext();
      assert dirIter.hasNext();

      ptrName = (String)ptrIter.next();

      // Handle pointer args
      size =(String)sizeIter.next();
      // *AP*  Will allow single core specification in the case of same core number
      // *AP*  This way core number specification can be more easily modified using sed
      if (coresIter.hasNext())
        cores = Integer.parseInt((String)coresIter.next());
      dir  = Integer.parseInt((String)dirIter.next());
      IDExpression ptrId = (IDExpression)new NameID(ptrName);
      VariableDeclaration ptrDecl = FCUDAutils.getVariableDeclarationUnchecked(ptrId, (SymbolTable)cStmt);
      if (ptrDecl == null) {
        System.out.println("Symbol "+ptrName+" is not a variable in transfer pragma.");
        System.exit(0);
      }

      //Check if pointer type or array type
      VariableDeclarator ptrDeclor = FCUDAutils.getDeclaratorFromDeclarationUnchecked(ptrDecl, ptrId);
      List<Specifier> ptrDeclorSpecs = ptrDeclor.getSpecifiers();
      if (ptrDeclorSpecs.size() == 0 || !(ptrDeclorSpecs.get(0) instanceof PointerSpecifier)) {
        ptrDeclorSpecs = ptrDeclor.getArraySpecifiers();
        if (ptrDeclorSpecs.size() == 0 || !(ptrDeclorSpecs.get(0) instanceof ArraySpecifier)) {
          System.out.println("Symbol "+ptrName+" in transfer pragma (representing off-chip memory) is neither pointer nor array type");
          System.exit(0);
        }
        // *TAN* I assume it is 1-dimensional array. Need to fix later for multi-dimensional array
        bound = Integer.parseInt(((ArraySpecifier)(ptrDeclorSpecs.get(0))).getDimension(0).toString());
      }

      leadSpecList.clear();
      // *TAN* temporarily disable VOLATILE
      //leadSpecList.add(Specifier.VOLATILE);
      leadSpecList.addAll(ptrDecl.getSpecifiers());
      VariableDeclaration volatPtrDecl = new VariableDeclaration(
         leadSpecList,
         (Declarator)new VariableDeclarator(
            ptrDeclor.getSpecifiers(),
            ptrId,
            ptrDeclor.getTrailingSpecifiers()));

      //Need to make this argument volatile
      // *TAN* temporarily disable VOLATILE
      //mToMakeVolatile.add(ptrDecl);

      if (!taskArgSet.contains((IDExpression)ptrId)) {
        taskDecls.add(volatPtrDecl);
        taskArgs.add(new Identifier(ptrDeclor));
        taskArgSet.add(new Identifier(ptrDeclor));
        commonArgsIndex.add(taskArgs.size()-1);
      }


      // Handle BRAM args
      IDExpression bramId = null;
      transferStmt = FCUDAutils.getNxtStmt(transferStmt);
      System.out.println("transferStmt: "+transferStmt.toString());
      bramId = findBRAM(ptrId, dir, transferStmt);

      // *AP* FIXME - Add special handling for NON-SHARED BRAMs, e.g. __constant__, __texture__, etc
      if (bramId == null) {
        System.out.println("Cannot handle NON-SHARED BRAMS currently");
        System.exit(0);
      }

      mTransfInfoSet.add(new TransferInfo(transferStmt, bramId, ptrId, dir, size, cores));
      VariableDeclaration bramDecl = (VariableDeclaration) SymbolTools.findSymbol((SymbolTable)cStmt, bramId);
      if (bramDecl == null) {
        System.out.println("Symbol "+bramId+" does not have declaration.");
        System.exit(0);
      }

      Expression sizeExpr = null;
      int sizeInt = 0;
      try {
        sizeInt = Integer.parseInt(size);
        sizeExpr = (Expression)(new IntegerLiteral(sizeInt));
      }
      catch(Exception e) {
        sizeExpr = (Expression)((IDExpression)new NameID(size));
        // Add identifier to task parameters if not global
        SymbolTable symT = FCUDAutils.getSymbolTable(annotStmt);
        VariableDeclaration sDecl = FCUDAutils.getVariableDeclarationUnchecked((IDExpression) sizeExpr, symT);
        assert (sDecl.getNumDeclarators() == 1);
        VariableDeclarator sDeclor = (VariableDeclarator)sDecl.getDeclarator(0);
        if (!SymbolTools.isGlobal((Symbol)sDeclor)) {
          if (!taskArgSet.contains((IDExpression) sizeExpr)) {
            taskDecls.add((VariableDeclaration) sDecl.clone());
            taskArgs.add(new Identifier(sDeclor));
            taskArgSet.add(new Identifier(sDeclor));
          }
        }
      }


      if(!taskArgSet.contains(bramId)) {
        taskArgs.add((IDExpression)bramId.clone());
        taskArgSet.add(bramId);
        taskDecls.add((Declaration)bramDecl.clone());
      }

      // Handle memcpy (burst) arguments
      memcpyArgs = new LinkedList();
      int bramDim = mBRAMSet.get(bramId);
      System.out.println("BRAM: "+bramId.toString()+"  Dim: "+bramDim);
      // *TAN* add offset to on-chip memory pointer
      leadSpecList.clear();
      leadSpecList.add(Specifier.INT);

      Expression onChipOffset = FCUDAGlobalData.analyzeAccess(transferStmt, bramId);
      String prefixOffset = taskName + "_" + bramId.toString() + "_offset";
      IDExpression coeffOffset = (IDExpression)new NameID(prefixOffset);
      VariableDeclarator offsetDeclor = new VariableDeclarator(coeffOffset);
      VariableDeclaration offsetDeclion = new VariableDeclaration(leadSpecList, (Declarator)offsetDeclor);

      proc.getBody().addANSIDeclaration((Declaration)offsetDeclion);
      if (onChipOffset != null)
        cStmt.addStatementBefore(
           annotStmt,
           new ExpressionStatement(
            new AssignmentExpression(
             new Identifier(offsetDeclor),
             AssignmentOperator.NORMAL,
             (Expression)onChipOffset.clone())));
      else
        cStmt.addStatementBefore(
           annotStmt,
           new ExpressionStatement(
            new AssignmentExpression(
             new Identifier(offsetDeclor),
             AssignmentOperator.NORMAL,
             new IntegerLiteral(0))));

      taskArgs.add(new Identifier(offsetDeclor));
      taskDecls.add((VariableDeclaration)offsetDeclion.clone());

      //Block ram id added for memcpy function
      if (bramDim > 1)
      {
        /**
         * TAN: We need to take into account of the value passed
         * to the argument "numDims" in command line when translating 
         * CUDA code. Basically it tells us the number of block
         * Dimension we want to translate (threadIdx.x, threadIdx.y,...)
         * So we need to check whether the value of "numDims" is
         * identical with the dimension of the BRAM or not.
         */
        LinkedList<Expression> bramArrayIdx = new LinkedList();
        numDims = FCUDAGlobalData.getKernTblkDim(mProcedure);
        ArrayAccess bramArg;
        if (numDims == bramDim) {
          bramArrayIdx.add(MCUDAUtils.Tidx.getId(bramDim - 1));
          bramArg = new ArrayAccess((IDExpression)bramId.clone(), bramArrayIdx);
        } else {
          // handle multi-dimensional BRAM array when numDims != bramDim
          Expression exp = (((ArrayAccess)bramId.getParent()).getIndex(0));
          LinkedList<Identifier> listIds = new LinkedList();
          if (exp instanceof Identifier) {
            Identifier id = (Identifier)exp;
            listIds.add(id);
          } else {
            for (Traversable t : exp.getChildren()) {
              if (t instanceof Identifier) {
                Identifier id = (Identifier)t;
                listIds.add(id);
              }
            }
          }
          bramArg = new ArrayAccess(bramId.clone(), 
              (((ArrayAccess)bramId.getParent()).getIndex(0)).clone());

          for (Identifier id: listIds) {
            VariableDeclarator idDeclor = new VariableDeclarator(id.clone());
            VariableDeclaration idDeclion = new VariableDeclaration(leadSpecList, (Declarator)idDeclor);
            taskArgs.add(new Identifier(idDeclor));
            taskDecls.add((VariableDeclaration)idDeclion.clone());
          }
        }
        memcpyArgs.add(Symbolic.add(bramArg, new Identifier(offsetDeclor)));
      }
      else
        memcpyArgs.add(Symbolic.add((IDExpression) bramId.clone(), new Identifier(offsetDeclor)));
      Expression[] affineCoeffs;

      // Check the address form - (See PrepareForSplit.java for details)
      if (baseList.size() == 0) {
        Expression baseAddrForBurst = FCUDAGlobalData.analyzeAccess(transferStmt, ptrDeclor.getID());
        if (baseAddrForBurst == null) {
          System.out.println("Could not handle ptr access for "+ptrId.toString());
          System.exit(0);
        }
        affineCoeffs =  FCUDAGlobalData.getBaseTerms(proc, ptrName);
      }
      else {
        affineCoeffs = new Expression[6];
        parseBaseTerms((String)baseIter.next(), affineCoeffs, (SymbolTable)cStmt);
      }

      Symbol X = null;
      Symbol c1 = null;

      for(int j=0; j<DataDepAnalysis.NUMBER_BASE_TERMS; ++j) {
        String prefix = "";
        if (j == 0)
          prefix = taskName+"_"+bramId.toString()+"_X";
        else
          prefix = taskName+"_"+bramId.toString()+"_c";
        IDExpression coeffVar = (IDExpression)new NameID(prefix+"_"+j);
        VariableDeclarator cDeclor = new VariableDeclarator(coeffVar);
        VariableDeclaration cDeclion = new VariableDeclaration(leadSpecList, (Declarator)cDeclor);

        if (j == 0)
          X = cDeclor;
        if (j == 1)
          c1 = cDeclor;

        proc.getBody().addANSIDeclaration((Declaration)cDeclion);
        if (affineCoeffs[j] != null)
          cStmt.addStatementBefore(
             annotStmt,
             new ExpressionStatement(
              new AssignmentExpression(
               new Identifier(cDeclor),
               AssignmentOperator.NORMAL,
               (Expression)affineCoeffs[j].clone())));
        else
          cStmt.addStatementBefore(
             annotStmt,
             new ExpressionStatement(
              new AssignmentExpression(
               new Identifier(cDeclor),
               AssignmentOperator.NORMAL,
               new IntegerLiteral(0))));

        taskArgs.add(new Identifier(cDeclor));
        taskDecls.add((VariableDeclaration)cDeclion.clone());
      }

      // Off chip memmory index (ptr+X+c1*threadIdx)
      memcpyArgs.add(
         Symbolic.add(
          new Identifier(ptrDeclor),
          Symbolic.add(
           new Identifier(X),
           Symbolic.multiply(
            new Identifier(c1),
            MCUDAUtils.Tidx.getId(1)))));

      //if (write to off-chip) reverse 1st and 2nd elements
      if(dir == 1) {
        Expression tmp = memcpyArgs.get(0);
        memcpyArgs.set(0, memcpyArgs.get(1));
        memcpyArgs.set(1, tmp);
      }

      //memcpy size
      Specifier typeSpec = FCUDAutils.getDataType(ptrDecl);
      LinkedList<Specifier> specListSizeOf = new LinkedList();
      specListSizeOf.add(typeSpec);
      Expression memcpySizeArg = new BinaryExpression
        (
         (Expression)sizeExpr.clone(),
         BinaryOperator.MULTIPLY,
         new SizeofExpression(specListSizeOf)
        );
      memcpyArgs.add(memcpySizeArg);
      //memcpy call
      memcpyCall = new FunctionCall(new NameID("memcpy"), memcpyArgs);

      // *AP* Create separate loop body for each transfer
      // *AP* mainly because different memories might need different partitioning
      // *AP* Threadloops could potentially be fused if no difference in partitioning
      if (bramDim > 1) {
        CompoundStatement threadloopBody = new CompoundStatement();
        threadloopBody.addStatement(new ExpressionStatement(memcpyCall));
        ThreadLoop burstloop = new ThreadLoop(threadloopBody, numDims);
        burstloop.setInit(0, null);
        //if(bramDim == 3)
          //burstloop.setInit(1, null);
        //else
        //if (bramDim == numDims)
          //burstloop.setInit(bramDim - 1, null);
        //else
          //burstloop.setInit(numDims - 1, null);
        fcTask.addStatement(burstloop);

      }
      else   // No threadloop for one dimensional BRAMS
        fcTask.addStatement(new ExpressionStatement(memcpyCall));
    }
    coresList.clear();
    sizeList.clear();
    dirList.clear();
    arrayDims.clear();
    leadSpecList.clear();
    trailSpecList.clear();

  } // addTransferParameters()


  private void splitTasks(Procedure proc)
  {
    System.out.println("fcudaCores (splitTasks-start):\n"+FCUDAGlobalData.getFcudaCores(proc).toString());
    System.out.println("coreNames: \n"+FCUDAGlobalData.getCoreNames().toString());

    BreadthFirstIterator iter = new BreadthFirstIterator(proc);
    iter.pruneOn(Expression.class);
    String curTask = null;
    FcudaAnnotation fcAnnot = null, taskAnnot=null, beginAnnot = null;
    int stmtLevel=0;
    int annotLevel=0;
    boolean inTask = false;
    CompoundStatement fcTask = null;
    Procedure fcProc;
    FunctionCall taskCall = null;
    Set<Expression> taskUses = null;
    Set<Expression> taskDefs = null;
    Set<Expression> taskMaydefs = null;

    String annotType = "";
    String unrollFactor = "";
    String mpartFactor = "";
    String splitArrays = "";
    String shapes = "";
    // Eric
    String shared      = "";
    String nonshared   = "";
    String bramcore    = "";

    while(iter.hasNext()) {
      Statement stmt = null;
      try {
        stmt = (Statement)iter.next(Statement.class);
      }
      catch (NoSuchElementException e) {
        break;
      }

      // Add statement into new procedure
      if (inTask) {
        stmtLevel = getBlockLevel(stmt,proc.getClass());

        if (!(stmt instanceof AnnotationStatement)) {
          FCUDAGlobalData.addAnnot2Stmts(fcAnnot, stmt);
        }

        System.out.println("Task stmt: "+stmt.toString());
        System.out.println("of type: "+ stmt.getClass().toString());
        taskUses.addAll(DataFlowTools.getUseSet(stmt));
        taskDefs.addAll(DataFlowTools.getDefSet(stmt));

        // Keep track of declarations that remain wrapped in task annotations
        // Their declarators should only be used within task (because task
        // annotations should not cross block levels)
      }

      if (!(stmt instanceof AnnotationStatement))
        continue;

      System.out.println("Checking Annotation Statement: "+ stmt.getAnnotations().toString());

      // Only interested in TRANSFER and COMPUTE FCUDA pragmas
      List<FcudaAnnotation> fcAnnots = stmt.getAnnotations(FcudaAnnotation.class);
      if (fcAnnots.isEmpty())
        continue;
      fcAnnot = fcAnnots.get(0);

      // Deal with other FCUDA pragma factors
      annotType    = (String) fcAnnot.get("fcuda");
      unrollFactor = (String) fcAnnot.get("unroll");
      mpartFactor  = (String) fcAnnot.get("mpart");
      splitArrays  = (String) fcAnnot.get("array_split");
      shapes       = (String) fcAnnot.get("shape");
      nonshared    = (String) fcAnnot.get("non_shared");
      bramcore     = (String) fcAnnot.get("bram_core");
      shared       = (String) fcAnnot.get("shared");

      if (!(annotType.equals("transfer") || annotType.equals("compute"))) {
        if (curTask != null) { // Found other pragma between Compute or Transfer pragmas
          System.out.println("[FCUDA - Error] Detected non-task pragma between task pragmas: " + fcAnnot.toString());
          System.exit(0);
        } else
          continue;
      }

      // FCUDA annotations is TRANSFER or COMPUTE
      if (((String)fcAnnot.get("begin")).equals("true"))
        beginAnnot = fcAnnot;

      if (inTask)
        System.out.println("FCUDA "+annotType+" begin="+fcAnnot.get("begin")+" level="+stmtLevel);
      else
        System.out.println("FCUDA "+annotType+" begin="+fcAnnot.get("begin")+" level="+ getBlockLevel(stmt,proc.getClass()));

      String task = (String) fcAnnot.get("name");
      if ((curTask != null &&
            (!((String)fcAnnot.get("begin")).equals("false") || !task.equals(curTask))) ||
          (curTask == null && ((String)fcAnnot.get("begin")).equals("false"))) {
        System.out.println("[FCUDA - Error] Detected strange pragma sequence: " + fcAnnot.toString());
        System.exit(0);
      }
      else {
        if (curTask != null) { // End of current task
          if (annotLevel != stmtLevel) {
            System.out.println("[FCUDA - Error] Detected task pragmas at different hierarchy levels: " + fcAnnot.toString());
            System.exit(0);
          }
        
          curTask = null;
          inTask = false;

          Tools.printlnStatus("Task use set: " + taskUses.toString(), 0);
          Tools.printlnStatus("Task def set: " + taskDefs.toString(), 0);
          Tools.printlnStatus("Task maydef set: " + taskMaydefs.toString(), 0);

          Set<Expression> usesNdefs = new TreeSet<Expression>();
          usesNdefs.addAll(taskUses);
          usesNdefs.addAll(taskDefs);

          // Build task variable lists
          HashSet<IDExpression> taskArgSet = new HashSet();
          LinkedList<IDExpression> taskArgs = new LinkedList();
          LinkedList<Declaration> taskDecls = new LinkedList();

          // Get arguments set during preProcessPass (i.e. enable, gDim, bDim, bIdx for COMPUTE)
          FCUDAGlobalData.getArgsInfo(beginAnnot, taskArgSet, taskArgs, taskDecls);
          List<Statement> defStmts = new LinkedList<Statement>();

          buildTaskVars(proc, usesNdefs, taskArgs, taskArgSet, taskDecls, defStmts,
              (Statement)taskCall.getParent());

          System.out.println("taskArgs: " + taskArgs.toString());
          System.out.println("taskDecls: " + taskDecls.toString());
          System.out.println("defStmts: " + defStmts.toString());

          // Copy definitions of variables dependent on threadIdx implicit vars to task procedure
          Iterator<Statement> dStmtIter = defStmts.iterator();
          CompoundStatement taskBody = FCUDAGlobalData.getTaskBody(beginAnnot);

          Statement nxtStmt = IRTools.getFirstNonDeclarationStatement(taskBody);
          while (dStmtIter.hasNext()) {
            Statement dStmt = (Statement)dStmtIter.next();
            if (nxtStmt == null)
              taskBody.addStatement((Statement)dStmt.clone());
            else
              taskBody.addStatementBefore(nxtStmt, (Statement)dStmt.clone());

            IDExpression defID = null;

            Set<Expression> defs = DataFlowTools.getDefSet(dStmt);
            Iterator<Expression> dIter = defs.iterator();
            try {
              defID = (IDExpression)(dIter.next());
            }
            catch (ClassCastException e) {
              System.out.println("[FCUDA - Error] Can not get IDExpression from definition of threadIdx based variable ");
              System.exit(0);
            }
            Declaration dcl = defID.findDeclaration();
            taskBody.addANSIDeclaration((Declaration) dcl.clone());
          }

          //Determine which BRAMs are part of arguments of FCUDA tasks - this information will be used in pipelining
          for (Expression currExpr : taskArgs) {
            if ((currExpr instanceof IDExpression) && FCUDAGlobalData.isBRAM((IDExpression)currExpr))
              FCUDAGlobalData.addInterfaceBRAM((IDExpression)currExpr, taskCall);
          }
          // Add arguments to task call
          List<Expression> argList = new LinkedList<Expression>();
          argList.addAll(taskArgs);
          taskCall.setArguments(argList);

          // Generate new task procedure
          List<Declaration> declList = new LinkedList<Declaration>();
          declList.addAll(taskDecls);
          ProcedureDeclarator fcProcDecl = new ProcedureDeclarator(
             new NameID(proc.getName().toString() + "_" + task),
             declList);

          fcProc = new Procedure(Specifier.VOID, fcProcDecl, FCUDAGlobalData.getBody(beginAnnot));
          fcProc.annotate(taskAnnot);
          TranslationUnit parent = (TranslationUnit)proc.getParent();
          parent.addDeclarationBefore(proc, fcProc);
          // Store tasks and associated pragma parameters
          FCUDAutils.addTaskMapping(proc.getSymbolName(), fcProc);
          FCUDAutils.setTaskType(fcProc, annotType);
          if (unrollFactor != null)
            FCUDAutils.setTaskUnroll(fcProc, unrollFactor);
          if (mpartFactor != null)
            FCUDAutils.setTaskMempart(fcProc, mpartFactor);
          if (splitArrays != null) {
            LinkedList<String> arrayList = new LinkedList();
            FCUDAutils.parseList(splitArrays, arrayList);
            FCUDAutils.setTaskSplitArray(fcProc, arrayList);
          }
          if (shapes !=null) {
            LinkedList<String> shapeList = new LinkedList();
            FCUDAutils.parseList(shapes, shapeList);
            FCUDAutils.setTaskShape(fcProc, shapeList);
          }

          //Eric
          if (nonshared != null){
            LinkedList<String> arrayList = new LinkedList();
            FCUDAutils.parseList(nonshared, arrayList);
            FCUDAutils.setTaskNonSharedArray(fcProc, arrayList);
          }

          if (bramcore != null) {
            LinkedList<String> arrayList = new LinkedList();
            FCUDAutils.parseList(bramcore, arrayList);
            FCUDAutils.setTaskBramCore(fcProc, arrayList);
          }

          if (shared != null) {
            LinkedList<String> arrayList = new LinkedList();
            FCUDAutils.parseList(shared, arrayList);
            FCUDAutils.setTaskSharedArray(fcProc, arrayList);
          }
        }
        else { 	// Beginning of new task
          curTask = new String(task);
          annotLevel = getBlockLevel(stmt, proc.getClass());
          inTask = true;

          // Generate task call
          taskCall = FCUDAGlobalData.getFcudaCore(beginAnnot);
          ExpressionStatement callExpr = new ExpressionStatement(taskCall);
          ((CompoundStatement)stmt.getParent()).addStatementAfter(stmt, (Statement)callExpr);

          fcTask = FCUDAGlobalData.getTaskBody(beginAnnot);
          taskAnnot = fcAnnot;

          taskUses = new TreeSet<Expression>();
          taskDefs = new TreeSet<Expression>();
          taskMaydefs = new TreeSet<Expression>();
        }
      }
    }
    doActualDetach();
  } // splitTasks()


  /** Populate: - mBRAMset:  set of __shared__ variables
   *            - mBRAMDecl: subset of mBRAMset variables which
   *                        are declared in deeper levels of the kernel
   */
  private void addAllSharedToBRAMSet(Procedure proc)
  {
    PostOrderIterator iter = new PostOrderIterator(proc);
    iter.pruneOn(Expression.class);

    while (iter.hasNext()) {
      DeclarationStatement currDeclStmt = null;
      try {
        currDeclStmt = (DeclarationStatement)iter.next(DeclarationStatement.class);
      }
      catch(NoSuchElementException e) {
        break;
      }

      Declaration decl = currDeclStmt.getDeclaration();
      if (decl instanceof VariableDeclaration) {
        VariableDeclaration vDecl = (VariableDeclaration)decl;
        HashSet<Specifier> specSet = new HashSet();
        specSet.addAll(vDecl.getSpecifiers());
        assert vDecl.getNumDeclarators() == 1;  // *AP* this should hold after pass SeparateInitializers
        if (specSet.contains(Specifier.SHARED)) {
          IDExpression idexp = vDecl.getDeclaredIDs().get(0);
          List<Specifier> specs = ((VariableDeclarator)vDecl.getDeclarator(0)).getArraySpecifiers();
          if (specs != null && !specs.isEmpty()) {
            int dimNum = ((ArraySpecifier) specs.get(0)).getNumDimensions();
            System.out.println("BRAM:"+idexp+"  specs: "+specs.toString()+" size:"+dimNum);
            mBRAMSet.put(idexp, dimNum);
          }

          // *AP* Check if BRAM is declared at procedure level
          // *AP* If not, mark declaration for future move to procedure level
          // *AP* (should allocate BRAMs only once at beginning of Procedure invocation)
          if (getBlockLevel(vDecl, proc.getClass()) > 1)
            mBRAMDecl.add((Declaration)vDecl);
        }
      }
    }

    // *TAN* get BRAM identifier from proc's parameters
    // *TAN* probably due to FixMemoryParams
    List<Declaration> list_params = proc.getDeclarator().getParameters();
    for (Declaration param : list_params) {
      VariableDeclaration paramDecl = (VariableDeclaration)param;
      HashSet<Specifier> param_specSet = new HashSet();
      param_specSet.addAll(paramDecl.getSpecifiers());
      assert paramDecl.getNumDeclarators() == 1;  // *AP* this should hold after pass SeparateInitializers
      if (param_specSet.contains(Specifier.SHARED)) {
        IDExpression param_exp = paramDecl.getDeclaredIDs().get(0);
        List<Specifier> param_specs = ((VariableDeclarator)paramDecl.getDeclarator(0)).getArraySpecifiers();
        if(param_specs != null && !param_specs.isEmpty()) {
          int param_dimNum = ((ArraySpecifier) param_specs.get(0)).getNumDimensions();
          System.out.println("BRAM:"+param_exp+"  specs: "+param_specs.toString()+" size:"+param_dimNum);
          mBRAMSet.put(param_exp, param_dimNum);
        }
        paramDecl.getSpecifiers().remove(Specifier.SHARED);
      }
    }
  }


  private void preProcessPass(Procedure proc)
  {
    // Populate mBRAMset and mBRAMDecl sets
    addAllSharedToBRAMSet(proc);

    boolean addBidxDeclaration = true;

    PostOrderIterator iter = new PostOrderIterator(proc);
    iter.pruneOn(Expression.class);
    FcudaAnnotation fcAnnot = null, transferAnnot = null;
    CompoundStatement fcTask = null, enableBlock = null;

    String annotType = "";
    FunctionCall taskCall = null;
    while (iter.hasNext()) {
      Statement stmt = null;
      try {
        stmt = (Statement)iter.next(Statement.class);
      }
      catch (NoSuchElementException e) {
        break;
      }
      if (!(stmt instanceof AnnotationStatement))
        continue;      // Proceed for Annotation Statements

      List<FcudaAnnotation> fcAnnots = stmt.getAnnotations(FcudaAnnotation.class);
      if (fcAnnots.isEmpty())
        continue;

      fcAnnot = (FcudaAnnotation)fcAnnots.get(0);
      System.out.println("\n ... Preprocessing pragma: \n\t"+fcAnnot.toString());
      annotType = (String) fcAnnot.get("fcuda");

      if (((String)fcAnnot.get("begin")).equals("true")) {
        // add annotation to mFcudaAnnotStmts List
        FCUDAGlobalData.addFcudaAnnotationStmt(stmt);
        Set<IDExpression> taskArgSet = new TreeSet<IDExpression>();
        // *KG* - taskArgs needs to be list as order of args arbitrary in Set
        LinkedList<IDExpression> taskArgs = new LinkedList<IDExpression>();
        LinkedList<Declaration> taskDecls = new LinkedList<Declaration>();
        LinkedList<Integer> commonArgsIndex = new LinkedList();

        taskArgs.clear();
        taskArgSet.clear();
        taskDecls.clear();
        commonArgsIndex.clear();

        fcTask = new CompoundStatement();
        enableBlock = new CompoundStatement();

        FCUDAGlobalData.addAnnot2Body(fcAnnot, fcTask);
        FCUDAGlobalData.addAnnot2TaskBody(fcAnnot, enableBlock);

        //FIXME: SerializeThreads pass wraps loop-nest around the enable signal :( CRITICAL!
        //Enable signal for this compute/tranfer core
        IDExpression enableSignal = (IDExpression)new NameID("enableSignal_"+fcAnnot.get("name"));
        VariableDeclarator vTmp =  new	VariableDeclarator(enableSignal);
        VariableDeclaration enableDecl = new VariableDeclaration(Specifier.INT,vTmp);

        proc.getBody().addDeclaration((Declaration)enableDecl);

        Expression enablecheck_x = new BinaryExpression(
           MCUDAUtils.getBidxID(0),
           BinaryOperator.COMPARE_LT,
           MCUDAUtils.getGdimID(0));

        Expression enablecheck_y = new BinaryExpression(
           MCUDAUtils.getBidxID(1),
           BinaryOperator.COMPARE_LT,
           MCUDAUtils.getGdimID(1));

        Expression enable_all = new BinaryExpression(
           enablecheck_x,
           BinaryOperator.LOGICAL_AND,
           enablecheck_y);

        // *TAN* if this is streaming transfer task
        // enableSignal is always true, provided that
        // ping-pong buffers opt. does not apply to streamming task
        String transferType = (String) fcAnnot.get("type");
        if (transferType == "stream") {
          enable_all = new IntegerLiteral(1);
          addBidxDeclaration = false;
        }

        FCUDAutils.addAfterLastDeclaration(
            proc.getBody(),
            new ExpressionStatement(
             new AssignmentExpression(
              new Identifier(vTmp),
              AssignmentOperator.NORMAL,
              enable_all)));

        taskCall = new FunctionCall(new NameID(proc.getSymbolName() + "_" + fcAnnot.get("name")));
        FCUDAGlobalData.addAnnot2FcudaCore(fcAnnot, taskCall);

        FCUDAGlobalData.addFcudaCore(proc, taskCall);
        FCUDAGlobalData.setCoreName(taskCall, (String)fcAnnot.get("name"));

        if (annotType.equals("transfer"))
          FCUDAGlobalData.setCoreType(taskCall, FcudaCoreData.TRANSFER_TYPE);
        else
          FCUDAGlobalData.setCoreType(taskCall, FcudaCoreData.COMPUTE_TYPE);

        // *AP* Store enableSignal as NameID and look for appropriate symbol every time you inject it in AST
        FCUDAGlobalData.addEnableSignal(taskCall, enableSignal);

        taskArgs.add(new Identifier(vTmp));
        taskArgSet.add(new Identifier(vTmp));
        VariableDeclaration enableDeclClone = enableDecl.clone();
        taskDecls.add(enableDeclClone);

        //Add blockIdx, blockDim and gridDim as parameters
        taskDecls.addAll(MCUDAUtils.Bdim.getDecl());
        taskDecls.addAll(MCUDAUtils.Gdim.getDecl());
        if (addBidxDeclaration && !annotType.equals("transfer"))
          taskDecls.addAll(MCUDAUtils.Bidx.getDecl());

        taskArgs.add((IDExpression)(MCUDAUtils.Bdim.getId().get(0)));
        commonArgsIndex.add(taskArgs.size()-1);
        taskArgs.add((IDExpression)(MCUDAUtils.Gdim.getId().get(0)));
        commonArgsIndex.add(taskArgs.size()-1);
        if (addBidxDeclaration && !annotType.equals("transfer"))
          taskArgs.add((IDExpression)(MCUDAUtils.Bidx.getId().get(0)));

        taskArgSet.add((IDExpression)(MCUDAUtils.Bdim.getId().get(0)));
        taskArgSet.add((IDExpression)(MCUDAUtils.Gdim.getId().get(0)));
        if (addBidxDeclaration && !annotType.equals("transfer"))
          taskArgSet.add((IDExpression)(MCUDAUtils.Bidx.getId().get(0)));

        IfStatement enableStmt = new IfStatement(
           new Identifier(
            (VariableDeclarator)enableDeclClone.getDeclarator(0)),
           enableBlock);
        fcTask.addStatement(enableStmt);
        if (annotType.equals("transfer")) {
          fcTask.addANSIDeclaration(MCUDAUtils.Tidx.getDecl().get(0));
          transferAnnot = fcAnnot;
          //*KG* - modify transfer functions to have arguments for burst etc
          addTransferParameters(proc, transferAnnot, taskDecls, taskArgs, taskArgSet,
              enableBlock, commonArgsIndex);
        }

        if (fcAnnot.get("cores") != null) {
          LinkedList<String> numCoresList = new LinkedList();
          FCUDAutils.parseList((String)fcAnnot.get("cores"), numCoresList);
          FCUDAGlobalData.setNumCores(taskCall, Integer.parseInt(numCoresList.get(0)));
        }
        FCUDAGlobalData.clearArgsInfo(fcAnnot);
        FCUDAGlobalData.addArgsInfo(fcAnnot, taskArgSet, taskArgs, taskDecls);
        FCUDAGlobalData.addCommonArgsIndex(taskCall, commonArgsIndex);
      }
    }

    if (addBidxDeclaration)
      proc.getBody().addDeclaration(MCUDAUtils.Bidx.getDecl().get(0));

    // Remove transfer statements after converting them to memcpy's (bursts)
    for(TransferInfo tInfo : mTransfInfoSet)
      tInfo.getStmt().detach();
    mTransfInfoSet.clear();

    // Move BRAM declarations to procedure level
    for(Declaration bramDecl : mBRAMDecl)
    {
      Statement declStmt = IRTools.getAncestorOfType(bramDecl, Statement.class);
      declStmt.detach();

      // *FIXME* Should check if declaration of identifier exists in procedure or other
      //         intermediate levels

      proc.getBody().addANSIDeclaration(bramDecl.clone());
    }

    FCUDAGlobalData.setBRAMSet(mBRAMSet);
  } // preProcessPass()


  public void doActualDetach()
  {
    FCUDAGlobalData.initAnnotIterator();
    while (true) {
      Map.Entry<FcudaAnnotation, AnnotData> t = FCUDAGlobalData.getNextAnnot();
      if (t == null)
        break;
      CompoundStatement fcTask = FCUDAGlobalData.getTaskBody(t.getKey());
      Iterator iter = ((AnnotData)t.getValue()).getStatementList().iterator();
      while (iter.hasNext()) {
        Statement stmt = (Statement)iter.next();
        stmt.detach();
        fcTask.addStatement(stmt);
      }
    }
  }

  private void makeVolatile(Procedure proc)
  {
    for (VariableDeclaration v : mToMakeVolatile)
      v.prependSpecifierUnchecked(Specifier.VOLATILE);
  }

  public void transformProcedure(Procedure proc)
  {
    mProcedure = proc;
    mTransfInfoSet.clear();
    mToMakeVolatile.clear();
    mBRAMSet.clear();
    mBRAMDecl.clear();

    preProcessPass(proc);
    splitTasks(proc);
    makeVolatile(proc);
  }



  /**
   * Get the block depth of the traversable object t in terms of
   * a provided class type
   * *TODO* This may need to be removed (may not be safe tactic??)
   */
  public static int getBlockLevel(Traversable t, Class c)
  {
    int level = 0;
    while (true) {
      if (t.getClass().equals(c)) break;
      t = t.getParent();
      if (t instanceof CompoundStatement)
        level++;
    }
    return level;
  }
}
