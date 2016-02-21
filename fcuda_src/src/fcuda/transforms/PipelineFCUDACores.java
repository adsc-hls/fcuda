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

import fcuda.common.*;
import fcuda.utils.*;
import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;

import java.io.*;

/**
 *
 * Pass to pipeline FCUDA cores - currently the way it is done is to go to the innermost loop
 * in any loop nest and pipeline consecutive FCUDA cores (super simplistic)
 *
 * Must be run after IfSplit pass as only while and if-else are the control flow exist after
 * if-split pass
 *
 */

public class PipelineFCUDACores extends KernelTransformPass
{
  private Procedure mProcedure;
  private LinkedList<WhileLoop> mInnerMostWhileLoops;
  private HashSet<Statement> mCoreStmts;
  private int mPingPongId;
  private int mLoopIndex;
  private boolean mIsParentBlockIdxLoop;

  private HashMap<String, VariableDeclaration> id2duplicateSym;
  private HashMap<WhileLoop, LinkedList<FunctionCall> > core2WhileLoop;
  private HashMap<FunctionCall, Integer> core2StageID;
  public PipelineFCUDACores(Program program)
  {
    super(program);
    mInnerMostWhileLoops = new LinkedList();
    mCoreStmts = new HashSet();
    mPingPongId = 0;
    mLoopIndex = 0;
    mIsParentBlockIdxLoop = false;
    id2duplicateSym = new HashMap<String, VariableDeclaration>();
    core2WhileLoop = new HashMap<WhileLoop, LinkedList<FunctionCall> >();
    core2StageID = new HashMap<FunctionCall, Integer>();
    clear();
  }

  public void clear()
  {
    mInnerMostWhileLoops.clear();
    mCoreStmts.clear();
    id2duplicateSym.clear();
    core2WhileLoop.clear();
    core2StageID.clear();
  }

  public String getPassName()
  {
    return new String("[PipelineFCUDACores-FCUDA]");
  }

  private boolean isParentLoopBlockIdxLoop()
  {
    return mIsParentBlockIdxLoop;
  }

  private void extendLoopExecution(Statement currLoopStmt, int numStages, Procedure proc)
  { 
    if (currLoopStmt instanceof WhileLoop && currLoopStmt.equals
        (FCUDAGlobalData.getBlockIdxLoop(proc)))
    {
      Expression loopExpr;
      IfStatement ifStmtBlkY = FCUDAGlobalData.getBlkYIf(proc);
      Expression newCond = new BinaryExpression(
         MCUDAUtils.getBidxID(1),
         BinaryOperator.COMPARE_GE,
         new BinaryExpression(
          MCUDAUtils.getGdimID(1),
          BinaryOperator.ADD,
          new BinaryExpression(
            new BinaryExpression(
              new IntegerLiteral(FCUDAGlobalData.getNumParallelThreadBlocks(mProcedure)*(numStages)),
              BinaryOperator.MULTIPLY,
              new NameID("num_cores")),
            BinaryOperator.ADD,
            new NameID("core_id"))));

      ifStmtBlkY.setControlExpression(newCond);
      CompoundStatement parentStmt = (CompoundStatement)(ifStmtBlkY.getParent());
      return;
    }

    // Extend for WhileLoop case
    if (currLoopStmt instanceof WhileLoop && FCUDAGlobalData.isCreatedWhileLoop((WhileLoop)currLoopStmt)) {
      WhileLoop whileLoop = (WhileLoop)currLoopStmt;
      IDExpression loopExpr = FCUDAGlobalData.getGuardVar(whileLoop);
      Expression lpExpr = IRTools.findExpression(mProcedure, loopExpr);
      IfStatement ifStmt = FCUDAGlobalData.getGuardIf	(whileLoop);
      CompoundStatement parentStmt = (CompoundStatement)(whileLoop.getParent());

      //Initialize to 0 - like leadCount and continue loop till pipeline is empty
      for (int i=1;i<numStages+1;++i) {
        ExpressionStatement initStmt = new ExpressionStatement(
           new AssignmentExpression(
            getDuplicateExpression(lpExpr, i),
            AssignmentOperator.NORMAL,
            new IntegerLiteral(0)));
        parentStmt.addStatementBefore(whileLoop, initStmt);
        Expression negExpr = Symbolic.negate(getDuplicateExpression(lpExpr, i));
        ifStmt.setControlExpression(Symbolic.and(
             ifStmt.getControlExpression(),
             negExpr));
      }

      //First time and last time should both be 0 for exit
      return;
    }

    System.out.println("Cannot handle loop-extension of statement "+currLoopStmt.toString());
    System.exit(0);

  } // extendLoopExecution()

  private void addStatementsBeforeParentLoop(CompoundStatement cStmt, LinkedList<Statement> stmtList)
  {
    Traversable t = (Traversable)cStmt;
    while (t != null && !(t instanceof ForLoop) && !(t instanceof WhileLoop))
      t = (Traversable)t.getParent();
    Statement loopStmt = null;
    if (t != null) {
      loopStmt = (Statement)t;
      t = (Traversable)t.getParent();
    }
    if (t == null || !(t instanceof CompoundStatement)) {
      System.out.println("No surrounding loop + CompoundStatement found for "+cStmt.toString()+"\n");
      System.exit(0);
    }
    CompoundStatement parent = (CompoundStatement)t;
    for (Statement currStmt : stmtList)
      parent.addStatementBefore(loopStmt, currStmt);
  }

  // Check whether a Statement is OK to move ahead of Pipelining
  private boolean canBeMovedBeforeStartOfPipeline(Statement stmt, HashSet<Expression> definedScalars)
  {
    if (stmt instanceof AnnotationStatement)
      return true;

    DepthFirstIterator dfsIter = new DepthFirstIterator(stmt);
    while(dfsIter.hasNext()) {
      Expression currExpr = null;
      try {
        currExpr = (Expression)dfsIter.next(Expression.class);
      }
      catch(NoSuchElementException e) {
        break;
      }
      if (currExpr instanceof ArrayAccess)
        return false;
      if (currExpr instanceof UnaryExpression &&
          ((UnaryExpression)currExpr).getOperator() == UnaryOperator.DEREFERENCE)
        return false;
    }

    //If the same variables has 2 or more defining stmts then cannot move to start of pipeline
    //FIXME: Solution SSA for scalar variables within loop body
    HashSet<Expression> tmpSet = new HashSet();
    tmpSet.clear();
    tmpSet.addAll(definedScalars);
    tmpSet.retainAll(DataFlowTools.getDefSet(stmt));
    if (tmpSet.size() > 0) {
      tmpSet.clear();
      return false;
    }
    return true;
  }

  //Identify all scalar variables defined in this compound stmt
  private void identifyDefinedScalars(CompoundStatement cStmt, HashSet<Expression> definedSet)
  {
    DepthFirstIterator dfsIter = new DepthFirstIterator(cStmt);
    Set<Expression> tmpSet;

    while (dfsIter.hasNext()) {
      Statement currStmt = null;
      try {
        currStmt = (Statement)dfsIter.next(Statement.class);
      } catch(NoSuchElementException e) {
        break;
      }

      tmpSet = DataFlowTools.getDefSet(currStmt);

      for (Expression currExpr : tmpSet) {
        if (FCUDAutils.isCudaDim3Struct(currExpr, "blockIdx")) {
          if (currExpr instanceof AccessExpression)
            definedSet.add(((AccessExpression)currExpr).getLHS());
          else
            if (currExpr instanceof IDExpression)
              definedSet.add(currExpr);
          continue;
        }
        if (currExpr instanceof IDExpression) {
          if (FCUDAutils.isScalar(FCUDAutils.getVariableDeclarator((IDExpression)currExpr)))
            definedSet.add(currExpr);
          continue;
        }
      }
    }
  }

  // Because of if-split pass all relevant while loops will be inside CompoundStatement or WhileLoop
  // Not inside any other control flow
  private int identifyInnerMostWhileLoops(LinkedList<WhileLoop> innerMostLoops, CompoundStatement cStmt)
  {
    FlatIterator flatIter = new FlatIterator(cStmt);
    int numInnerMostLoops = 0, tmpLoops = 0;
    while (flatIter.hasNext()) {
      Statement currStmt = null;
      try {
        currStmt = (Statement)flatIter.next(Statement.class);
      }
      catch (NoSuchElementException e) {
        break;
      }

      if (currStmt instanceof CompoundStatement)
        numInnerMostLoops += identifyInnerMostWhileLoops(innerMostLoops, (CompoundStatement)currStmt);

      if (currStmt instanceof ForLoop)
        numInnerMostLoops += identifyInnerMostWhileLoops(innerMostLoops, (CompoundStatement)
            (((ForLoop)currStmt).getBody()));

      if (currStmt instanceof WhileLoop) {
        WhileLoop currChildLoop = (WhileLoop)currStmt;

        tmpLoops = identifyInnerMostWhileLoops(innerMostLoops, (CompoundStatement)currChildLoop.getBody());
        if (tmpLoops == 0) {
          innerMostLoops.add(currChildLoop);
          ++numInnerMostLoops;
        }
        else
          numInnerMostLoops += tmpLoops;


      }
    }

    return numInnerMostLoops;
  }

  private VariableDeclaration getDuplicateDeclaration(Expression origExpr, int pipelineStage)
  {
    IDExpression dupId = null;
    IDExpression tmpId = null;
    VariableDeclaration dupDecl = null;

    if (origExpr instanceof IDExpression) {
      tmpId = (IDExpression)origExpr;
    } else if(origExpr instanceof AccessExpression) {
      tmpId = (IDExpression)(((AccessExpression)origExpr).getLHS());
    } else {
      System.out.println("Unsupported expression: "+origExpr.toString());
      System.exit(0);
    }

    dupId =  new NameID(tmpId.toString() + "_pipe_" + pipelineStage);

    if (!id2duplicateSym.containsKey(dupId.getName())) {
      dupDecl = FCUDAutils.duplicateDeclaration(tmpId, dupId, (SymbolTable)mProcedure.getBody());
      id2duplicateSym.put(dupId.getName(), dupDecl);
    } else
      dupDecl = id2duplicateSym.get(dupId.getName());
    return dupDecl;
  }

  private Expression getDuplicateExpression(Expression origExpr, int pipelineStage)
  {
    IDExpression dupId = null;
    IDExpression tmpId = null;
    Expression cloneExpr = origExpr.clone();
    VariableDeclaration dupDecl = getDuplicateDeclaration(origExpr, pipelineStage);
    if (origExpr instanceof IDExpression) {
      tmpId = (IDExpression)origExpr;
    } else if(origExpr instanceof AccessExpression) {
      tmpId = (IDExpression)(((AccessExpression)origExpr).getLHS());
    } else {
      System.out.println("Unsupported expression: "+origExpr.toString());
      System.exit(0);
    }

    dupId = new Identifier((VariableDeclarator)dupDecl.getDeclarator(0));

    if (origExpr instanceof IDExpression)
      cloneExpr = dupId;
    else
      IRTools.replaceAll(cloneExpr, tmpId, dupId);
    return cloneExpr;
  }

  private void insertDuplicatedVariables(HashMap<Expression, FunctionCall> duplicateVariables,
      int numStages, Statement endOfPipeline, LinkedList<Statement> addBeforeParentLoop)
  {
    CompoundStatement cStmt = (CompoundStatement)(endOfPipeline.getParent());
    Statement lastStmt = endOfPipeline;
    for (Expression currExpr : duplicateVariables.keySet()) {
      // Add Declaration's
      FunctionCall core = duplicateVariables.get(currExpr);
      for (int i = 1; i <= core2StageID.get(core); i++) {
        mProcedure.getBody().addANSIDeclaration(getDuplicateDeclaration(currExpr, i));
        }
      // dim3 blockIdx
      if (currExpr.toString().equals(MCUDAUtils.getBidxID().get(0).toString())) {
        ExpressionStatement blkIdxInitStmt = new ExpressionStatement(
            new AssignmentExpression(
              getDuplicateExpression(MCUDAUtils.getBidxID(0), core2StageID.get(core)),
              AssignmentOperator.NORMAL,
              new IntegerLiteral(-1)));
        addBeforeParentLoop.add(blkIdxInitStmt);

      }

      //Add update assignments
      for (int i = core2StageID.get(duplicateVariables.get(currExpr)); i >= 1; i--) {
        Expression prevExpr = null;
        if (i > 1)
          prevExpr = getDuplicateExpression(currExpr, i-1);
        else
          prevExpr = (Expression)currExpr.clone();
        Statement newStmt = new ExpressionStatement(
           new AssignmentExpression(
            getDuplicateExpression(currExpr, i),
            AssignmentOperator.NORMAL,
            prevExpr));
        cStmt.addStatementAfter(lastStmt, newStmt);
        lastStmt = newStmt;
        }
      }
  } // insertDuplicatedVariables()

  //
  //Will keep improving as we keep refining FCUDA
  //
  // More formal model for pipelining tasks?
  // Checks to add
  //
  //1. No interface array should be accessed by any statement other than inside the FCUDA cores
  //   within the parent compound statement of the pipelining section
  //2. All accesses to interface BRAMs are in consecutive FCUDA cores
  //	Later if not continuous then generate copy code for correct pipelining
  //3. Should generate prologue and epilogue code for correct operation - mainly involves data transfer
  //   ping-pong buffers <-> normal arrays outside the pipeline
  private boolean pipelineCheck(LinkedList<ExpressionStatement> pipeStmts, CompoundStatement cStmt)
  {
    if (pipeStmts.size() < 2) {
      System.out.println("Cannot perform loop unrolling:\n" + cStmt.toString());
      return false;
    }

    //Currently limited to the case where every access to an array is limited to 2 FCUDA cores only
    //then can simply use ping-pong else would need to use ping-pong-pang-peng-pung etc ..
    HashSet<FunctionCall> pipeFnCalls = new HashSet();
    for(ExpressionStatement currStmt : pipeStmts) {
      FunctionCall currCall = (FunctionCall)(currStmt.getExpression());
      pipeFnCalls.add(currCall);
    }

    //Check whether all BRAMs in interface of each call are accessed only by one previous or one next call
    Iterator iter = pipeStmts.iterator();
    FunctionCall prevCall = null;
    FunctionCall currCall = (FunctionCall)((ExpressionStatement)iter.next()).getExpression();
    FunctionCall nextCall = null;

    HashSet<FunctionCall> tmpSet = new HashSet();
    HashSet<FunctionCall> otherCores = new HashSet();
    otherCores.clear();

    int count = 1;
    while (count < pipeStmts.size() + 1) {
      if (count < pipeStmts.size())
        nextCall = (FunctionCall)((ExpressionStatement)iter.next()).getExpression();
      else
        nextCall = null;
      ++count;

      tmpSet.clear();
      tmpSet.addAll(pipeFnCalls);

      if (prevCall != null)
        tmpSet.remove(prevCall);
      tmpSet.remove(currCall);
      if (nextCall != null)
        tmpSet.remove(nextCall);

      List<Expression> argList = currCall.getArguments();
      for (Expression currExpr : argList) {
        otherCores.clear();
        if (currExpr instanceof IDExpression && FCUDAGlobalData.isInterfaceBRAM(
              (IDExpression)currExpr)) {
          otherCores.addAll(FCUDAGlobalData.getFCUDACoresForBRAM((IDExpression)currExpr));
          otherCores.retainAll(pipeFnCalls);

          if (otherCores.size() > 2) {
            System.out.println("Unfortunately pipelining is not possible because "
                +"more than 2 cores access BRAM "+currExpr.toString()+ " "
                +otherCores.toString());
            return false;
          }
          tmpSet.retainAll(otherCores);
        }
      }

      prevCall = currCall;
      currCall = nextCall;
    }

    otherCores.clear();
    System.out.println("Here is the set of functions for pipelining: \n");
    for (ExpressionStatement currStmt : pipeStmts)
      System.out.println(currStmt.toString());
    return true;
  } // pipelineCheck()


  private void handleBRAMs(LinkedList<ExpressionStatement> pipeStmts, CompoundStatement cStmt, int stageId,
          HashMap<String, FunctionCall> bramName2FirstDef,
          HashSet<String> declaredSet, HashMap<Expression, FunctionCall> duplicateSet,
          LinkedList<Statement> addBeforeParentLoop)
  {
    int index = 0;

    HashSet<Expression> tmpIntersectSet = new HashSet();

    HashSet<IDExpression> oldBRAMset = new HashSet();
    oldBRAMset.clear();

    HashMap<IDExpression, IDExpression> replaceInterfaceBRAMs = new HashMap();
    replaceInterfaceBRAMs.clear();

    HashMap<Expression, Expression> scalarDuplicate = new HashMap();
    scalarDuplicate.clear();

    HashSet<FunctionCall> tmpSet = new HashSet();
    tmpSet.clear();

    HashSet<FunctionCall> pipeCalls = new HashSet();
    pipeCalls.clear();

    for (ExpressionStatement currStmt : pipeStmts) {
      FunctionCall currCall = (FunctionCall)(currStmt.getExpression());
      pipeCalls.add(currCall);
    }

    FlatIterator flatIter = new FlatIterator(cStmt);
    FunctionCall prevCall = null;
    int pipe_stage = 0;
    while (flatIter.hasNext()) {
      Statement currStmt = null;
      try {
        currStmt = (Statement)flatIter.next(Statement.class);
      } catch(NoSuchElementException e) {
        break;
      }
      if (!(currStmt instanceof ExpressionStatement))
        continue;
      ExpressionStatement exprStmt = (ExpressionStatement)currStmt;
      if (!(exprStmt.getExpression() instanceof FunctionCall))
        continue;
      FunctionCall coreCall = (FunctionCall)exprStmt.getExpression();
      FunctionCall origCall = ((FunctionCall)pipeStmts.get(index).getExpression());
      //FIXME: Not perfect check, but OK
      // Check that coreCall is clone of origCall
      if (!origCall.equals(coreCall))
        continue;

      System.out.println(" *** coreCall == origCall");
     
      pipe_stage = core2StageID.get(origCall);

      prevCall = coreCall;
      replaceInterfaceBRAMs.clear();
      scalarDuplicate.clear();

      List<Expression> argList = origCall.getArguments();

      for (Expression currExpr : argList) {
        if (currExpr instanceof IDExpression && FCUDAGlobalData.isInterfaceBRAM((IDExpression)currExpr)) {
          tmpSet.clear();
          tmpSet.addAll(FCUDAGlobalData.getFCUDACoresForBRAM((IDExpression)currExpr));
          tmpSet.retainAll(pipeCalls);
          //If the argument is interface BRAM and it occurs in interface of at least 2 FCUDA tasks,
          //then introduce ping-pong buffers
          if (tmpSet.size() > 1) {
            String newName = "";
            IDExpression bramId = (IDExpression)currExpr;
            if ((stageId == 0 && origCall.equals(bramName2FirstDef.get(bramId.toString()))) ||
                (stageId == 1 && !origCall.equals(bramName2FirstDef.get(bramId.toString()))))
              newName = bramId.toString()+"_ping";
            else
              newName = bramId.toString()+"_pong";
            IDExpression newBramId = new NameID(newName);
            VariableDeclaration vDecl = null;
            if (!declaredSet.contains(newName)) {
              declaredSet.add(newName);
              vDecl = FCUDAutils.duplicateDeclaration(
                 bramId,
                 newBramId,
                 (SymbolTable)mProcedure.getBody());
              mProcedure.getBody().addANSIDeclaration(vDecl);
              oldBRAMset.add(bramId);
            }

            //Store for future replace
            vDecl = (VariableDeclaration)SymbolTools.findSymbol(
               (SymbolTable) mProcedure.getBody(),
               newBramId);
            replaceInterfaceBRAMs.put(
               bramId,
               new Identifier((VariableDeclarator)vDecl.getDeclarator(0)));
          }
        }

        //Scalar variables that needs to be duplicated to maintain correctness during pipelining

        //Is idExpr or AccessExpr (dim3 vars) and is defined in this compound stmt
        if ((currExpr instanceof IDExpression || currExpr instanceof AccessExpression)
            && duplicateSet.containsKey(currExpr))
          scalarDuplicate.put(currExpr, getDuplicateExpression(currExpr, pipe_stage));
        tmpIntersectSet.clear();
        tmpIntersectSet.addAll(DataFlowTools.getUseSet(currExpr));
        tmpIntersectSet.retainAll(duplicateSet.keySet());
        for (Expression tmpExpr : tmpIntersectSet) {
          scalarDuplicate.put(tmpExpr, getDuplicateExpression(tmpExpr, pipe_stage));
        }
        tmpIntersectSet.clear();
      }

      //do actual replace
      for (Map.Entry<IDExpression, IDExpression> currMapEntry : replaceInterfaceBRAMs.entrySet())
        Tools.replaceAll((Traversable)coreCall, currMapEntry.getKey(), currMapEntry.getValue());

      // init enable_signal vars to zero to avoid initial execution of cores
      // which may lead to segfault
      Expression enableSignal = FCUDAGlobalData.getEnableExpression(coreCall);
      Expression enableSignal_pipe = scalarDuplicate.get(enableSignal);
      if (enableSignal_pipe != null && stageId == 0) {
        for (int i = 1; i <= core2StageID.get(origCall); i++) {
          Statement initEnableSignal_pipe = new ExpressionStatement(
              new AssignmentExpression(
                getDuplicateExpression(enableSignal, i),
                AssignmentOperator.NORMAL,
                new IntegerLiteral(0)));
          addBeforeParentLoop.add(initEnableSignal_pipe);
        }
      }

      // Keep original vars for stage 0 of pipeline
      //if(index != 0)
      if (pipe_stage != 0)
        for(Map.Entry<Expression, Expression> currMapEntry : scalarDuplicate.entrySet())
          Tools.replaceAll((Traversable)coreCall, currMapEntry.getKey(), currMapEntry.getValue());

      FCUDAGlobalData.copyFcudaCoreData(origCall, coreCall);
      ++index;
    }

      // Remove old BRAM declarations
    for (IDExpression bID : oldBRAMset) {
      Declaration bDecl = bID.findDeclaration();
      Statement bDeclStmt = IRTools.getAncestorOfType((Traversable)bDecl, Statement.class);
      mProcedure.getBody().removeChild((Traversable)bDeclStmt);
    }

    replaceInterfaceBRAMs.clear();
    pipeCalls.clear();
    tmpSet.clear();
    scalarDuplicate.clear();
  } //handleBRAMs()


  private Statement pipelineCores(LinkedList<ExpressionStatement> pipeStmts, CompoundStatement cStmt,
      LinkedList<Statement> addBeforeParentLoop,
      HashMap<Expression, FunctionCall> duplicateSet,
      WhileLoop parentWhileLoop)
  {
    if (!pipelineCheck(pipeStmts, cStmt)) {
      System.out.println("NOT pipelining current block of statements"+cStmt.toString());
      return null;
    }

    //Set for all new declared BRAM arrays
    HashSet<String> newBRAMArrays = new HashSet();
    newBRAMArrays.clear();

    //Build map from bram to which function first "defines" it - really simple again
    //Assumes that the first function which has the array as the argument defines it
    // *AP* *FIXME* Invalid assumption
    HashMap<String, FunctionCall> bramName2FirstDef = new HashMap();
    bramName2FirstDef.clear();
    for (ExpressionStatement exprStmt : pipeStmts) {
      FunctionCall coreCall = (FunctionCall)exprStmt.getExpression();
      List<Expression> argList = coreCall.getArguments();
      for (Expression currExpr : argList)
        if (currExpr instanceof IDExpression &&
            FCUDAGlobalData.isInterfaceBRAM((IDExpression)currExpr) &&
            !bramName2FirstDef.containsKey(((IDExpression)currExpr).toString()))
          bramName2FirstDef.put((((IDExpression)currExpr).toString()), coreCall);
    }

    LinkedList<CompoundStatement> pingpongCStmts = new LinkedList();
    pingpongCStmts.clear();

    //Create ping-pong control variable
    IDExpression pingpongVar = new NameID("pingpong_"+mPingPongId++);
    VariableDeclarator ppDeclor = new VariableDeclarator(pingpongVar);
    VariableDeclaration varDecl = new VariableDeclaration(Specifier.INT, ppDeclor);
    mProcedure.getBody().addANSIDeclaration(varDecl);
    FCUDAGlobalData.addIgnoreDuringDuplication(FCUDAutils.getClosestParentStmt(varDecl));
    //Add initialization of the ping-pong control to parent
    Statement initPingPongStmt = new ExpressionStatement(
         new AssignmentExpression(
          new Identifier(ppDeclor),
          AssignmentOperator.NORMAL,
          new IntegerLiteral(0)));

    // Start here, collect new pipeline enable signals
    addBeforeParentLoop.add(initPingPongStmt);
    FCUDAGlobalData.addIgnoreDuringDuplication(initPingPongStmt);

    System.out.println("Before handleBRAMs coreNames: \n"+FCUDAGlobalData.getCoreNames().toString());

    FCUDAGlobalData.prepareToResetFCUDACores();
    //Can only handle ping-pong pipeline right now
    int newVal = 0;
    for (int i=0;i<2;++i) {
      CompoundStatement newCStmt = (CompoundStatement)cStmt.clone();
      handleBRAMs(pipeStmts, newCStmt, i, bramName2FirstDef, newBRAMArrays, duplicateSet, addBeforeParentLoop);
      if (i == 0)
        newVal = 1;
      else
        newVal = 0;
      Statement flipStmt = new ExpressionStatement(
         new AssignmentExpression(
          new Identifier(ppDeclor),
          AssignmentOperator.NORMAL,
          new IntegerLiteral(newVal)));
      newCStmt.addStatement(flipStmt);
      FCUDAGlobalData.addIgnoreDuringDuplication(flipStmt);
      pingpongCStmts.add(newCStmt);
    }

    System.out.println("After handleBRAMs coreNames: \n"+FCUDAGlobalData.getCoreNames().toString());

    FCUDAGlobalData.updateFCUDACores(mProcedure);

    System.out.println("After update coreNames: \n"+FCUDAGlobalData.getCoreNames().toString());

    IfStatement ifStmt = new IfStatement(
       Symbolic.eq(new Identifier(ppDeclor), new IntegerLiteral(0)),
       pingpongCStmts.get(0),
       pingpongCStmts.get(1));
    IfStatement ifStmt_clone = ifStmt.clone();
    FCUDAGlobalData.addPipelineCreated(ifStmt_clone);
    bramName2FirstDef.clear();
    newBRAMArrays.clear();

    // *TAN* Create a PipelineBlock CompoundStatement
    if (isParentLoopBlockIdxLoop()) {
      CompoundStatement pipelineBlock = new CompoundStatement();
      pipelineBlock.addStatement(ifStmt_clone);
      if (!mInnerMostWhileLoops.isEmpty())
        for (WhileLoop innerLoop : mInnerMostWhileLoops) {
          // Modify the WhileLoop to use correct pipeline stage
          LinkedList<FunctionCall> listCores = core2WhileLoop.get(innerLoop);
          if (listCores == null)
            continue;
          for (FunctionCall core : listCores) {
            List<Expression> argList = core.getArguments();
            LinkedList<Expression>arg2Duplicate = new LinkedList<Expression>();
            LinkedList<FunctionCall> otherCores = new LinkedList<FunctionCall>();

            HashMap<Expression, Expression> scalarDuplicate = new HashMap<Expression, Expression>();
            Expression enableSignal = null;
            for (Expression currExpr : argList) {
              if (currExpr instanceof IDExpression &&
                  FCUDAGlobalData.isInterfaceBRAM((IDExpression)currExpr)) {
                otherCores.addAll(FCUDAGlobalData.getFCUDACoresForBRAM((IDExpression)currExpr));
                otherCores.remove(core);
              }
              // dim3 var (blockIdx)
              if ((currExpr instanceof IDExpression || currExpr instanceof AccessExpression)
                  && duplicateSet.containsKey(currExpr)) {
                arg2Duplicate.add(currExpr);
              }
              if (currExpr.equals(FCUDAGlobalData.getEnableExpression(core))) {
                enableSignal = ((BinaryExpression)currExpr).getLHS();
                arg2Duplicate.add(enableSignal);
              }
            }

            if (!otherCores.isEmpty()) {
              // The very first core which uses the same BRAM as
              // the processing core. We need to pipeline the
              // processing core in the exact stage with this core.
              FunctionCall prevCore = otherCores.get(0);
              for (Expression currExpr : Tools.getUseSet(innerLoop)) {
                // dim3 var (blockIdx)
                if (currExpr instanceof AccessExpression)
                  currExpr = ((AccessExpression)currExpr).getLHS();
                FunctionCall call = duplicateSet.get(currExpr);
                // if there is a core which uses this expression
                // then duplicate (pipeline) it so the processing core
                // can use the "correct version" of this expression. 
                // Otherwise, no need
                if (call != null) {
                    arg2Duplicate.add(currExpr);
                }
              }
              int stageID = core2StageID.get(prevCore);
              FunctionCall origCore = core.clone();
              core2StageID.put(origCore.clone(), stageID);
              for (Expression arg : arg2Duplicate) {
                if (duplicateSet.containsKey(arg))
                  if (core2StageID.get(duplicateSet.get(arg)) >= stageID)
                    continue;
                duplicateSet.put(arg, origCore);
              }
              for (Expression arg : arg2Duplicate)
                scalarDuplicate.put(arg, getDuplicateExpression(arg, stageID));
              Expression enableSignal_pipe = scalarDuplicate.get(enableSignal);
              if (enableSignal_pipe != null) {
                for (int i = 1; i <= stageID; i++) {
                  Statement initEnableSignal_pipe = new ExpressionStatement(
                      new AssignmentExpression(
                        getDuplicateExpression(enableSignal, i),
                        AssignmentOperator.NORMAL,
                        new IntegerLiteral(0)));
                  addBeforeParentLoop.add(initEnableSignal_pipe);
                }
              }
              for(Map.Entry<Expression, Expression> currMapEntry : scalarDuplicate.entrySet())
                Tools.replaceAll((Traversable)innerLoop, currMapEntry.getKey(), currMapEntry.getValue());
            }

          }
          WhileLoop newLoop = innerLoop.clone();
          Statement guardInit = FCUDAGlobalData.getGuardInitStmt(innerLoop);
          Statement newGuardInit = guardInit.clone();
          // FIXME: I need to handle Comma Expression, later
          Statement loopVarInit = FCUDAGlobalData.getLoopVarInitStmt(innerLoop);
          Statement newLoopVarInit = loopVarInit.clone();
          IfStatement guardIf = FCUDAGlobalData.getGuardIf(innerLoop).clone();
          IDExpression guardVar = FCUDAGlobalData.getGuardVar(innerLoop).clone();

          FCUDAGlobalData.addWhileLoopData(newLoop, guardVar, guardIf);
          FCUDAGlobalData.setGuardInitStmt(newLoop, newGuardInit);
          FCUDAGlobalData.setLoopVarInitStmt(newLoop, newLoopVarInit);

          pipelineBlock.addStatement(newGuardInit);
          pipelineBlock.addStatement(newLoopVarInit);
          pipelineBlock.addStatement(newLoop);
          guardInit.detach();
          loopVarInit.detach();
          innerLoop.detach();
        }
      return pipelineBlock; //ifStmt;
    }
    else
      return ifStmt_clone;
  } // pipelineCores()

  // Really simple pipeline function: if all FCUDA cores in this compound statement are consecutive statements
  // then can pipeline else cannot
  private int pipelineFCUDAKernels(CompoundStatement cStmt, WhileLoop parentWhileLoop)
  {
    CompoundStatement newPipeCompoundStmt = new CompoundStatement();
    FlatIterator flatIter = new FlatIterator(cStmt);

    LinkedList<ExpressionStatement> pipeStmts = new LinkedList();
    LinkedList<Statement> detachStmts = new LinkedList();
    detachStmts.clear();
    pipeStmts.clear();

    int startState = 0;

    Statement endOfPipeline = null;
    Statement startOfPipeline = null;

    LinkedList<Statement> tmpStartOfPipeline = new LinkedList();
    LinkedList<Statement> moveToStartOfPipeline = new LinkedList();
    HashSet<Expression> definedScalars = new HashSet();
    definedScalars.clear();
    moveToStartOfPipeline.clear();
    tmpStartOfPipeline.clear();

    while (flatIter.hasNext()) {
      Statement currStmt = null;
      try {
        currStmt = (Statement)flatIter.next(Statement.class);
      } catch(NoSuchElementException e) {
        break;
      }

      Statement cloneStmt = (Statement)currStmt.clone();

      if (!(currStmt instanceof ExpressionStatement) ||
          !(mCoreStmts.contains(currStmt))) {
        //Ignore AnnotationStmts
        if (startState == 1 && !(currStmt instanceof AnnotationStatement)) {
          if (!canBeMovedBeforeStartOfPipeline(currStmt, definedScalars))
            startState = 2; // Found unmovable statements
          else
            tmpStartOfPipeline.add(currStmt);
        }
        definedScalars.addAll(DataFlowTools.getDefSet(currStmt));
        continue;
      }

      if (startState == 2) {
        System.out.println("Detected unmoveable statements between FCUDA cores \n"
            +pipeStmts.get(pipeStmts.size()-1)
            +"\n"+currStmt.toString());
        System.out.println("Could not pipeline FCUDA cores in : \n"+cStmt.toString());
        pipeStmts.clear();
        break;
      }

          // Here we handle Task calls (Fcuda cores)
      if(startState == 0)
        startOfPipeline = currStmt;

      moveToStartOfPipeline.addAll(tmpStartOfPipeline);
      tmpStartOfPipeline.clear();

      pipeStmts.add((ExpressionStatement)currStmt);
      endOfPipeline = currStmt;
      newPipeCompoundStmt.addStatement(cloneStmt);
      detachStmts.add(currStmt);
      startState = 1;
    }

    if (parentWhileLoop != null)
      for (ExpressionStatement stmt : pipeStmts) {
        LinkedList<FunctionCall> listCores = new LinkedList<FunctionCall>();
        FunctionCall core = (FunctionCall)(stmt.getExpression());
        if (core instanceof FunctionCall)
          listCores.add(core);
        core2WhileLoop.put(parentWhileLoop, listCores);
      }

    //Need at least 2 or more cores for pipelining
    int index = 0;
    if (pipeStmts.size() >= 2) {
      for (Statement currStmt : moveToStartOfPipeline) {
        currStmt.detach();
        cStmt.addStatementBefore(startOfPipeline, currStmt);
      }

      HashSet<Expression> definedSet = new HashSet();
      //Find all scalars defined in this compound statement
      identifyDefinedScalars(cStmt, definedSet);
      //Find all scalars which are used in (stages > first) of the pipeline and duplicate them
      HashMap<Expression, FunctionCall> duplicateSet = new HashMap();
      duplicateSet.clear();

      // Find all the args of Cores that need to be duplicated in different pipeline stages
      FunctionCall prevCall = null;
      for (ExpressionStatement tmpStmt : pipeStmts) {
        FunctionCall tmpCall = (FunctionCall)(tmpStmt.getExpression());
        // *TAN* if we have two consecutive transfer tasks --> they are in the same pipelined stage
        if (prevCall != null) {
          if (!(FCUDAGlobalData.getCoreType(tmpCall) ==
              FCUDAGlobalData.getCoreType(prevCall) && FCUDAGlobalData.getCoreType(tmpCall) == 2)) {

            List<Expression> curCallArgList = tmpCall.getArguments();
            List<Expression> prevCallArgList = prevCall.getArguments();

            // if the current core and the previous core do not have common BRAM
            // they should be in the same pipeline stages (for data flow to make sense)
            curCallArgList.retainAll(prevCallArgList);
            for (Expression currExpr : curCallArgList)
              if (currExpr instanceof IDExpression && FCUDAGlobalData.isInterfaceBRAM(
                  (IDExpression)currExpr)) {
                index++;
                break;
              }
          }
        }

        core2StageID.put(tmpCall, index);

        prevCall = tmpCall;
        //Expression enableExpr = FCUDAGlobalData.getEnableExpression(tmpCall);
        List<Expression> argList = tmpCall.getArguments();
        if (index > 0)
        for (Expression tmpArg : argList) {
          //Ignore the enable expression
          //if(!tmpArg.equals(enableExpr))
          {
            HashSet<Expression> tmpIntersectSet = new HashSet();
            tmpIntersectSet.clear();
            tmpIntersectSet.addAll(definedSet);
            tmpIntersectSet.retainAll(DataFlowTools.getUseSet(tmpArg));
            //Is idExpr or AccessExpr (dim3 vars) and is defined in this compound stmt
            if((tmpArg instanceof IDExpression || tmpArg instanceof AccessExpression)
                && definedSet.contains(tmpArg))
              duplicateSet.put(tmpArg, tmpCall);
            for (Expression tmp : tmpIntersectSet)
              duplicateSet.put(tmp, tmpCall);
            tmpIntersectSet.clear();
          }
        }
      }
      LinkedList<Statement> addBeforeParentLoop = new LinkedList();
      addBeforeParentLoop.clear();

      Statement insertStmt = pipelineCores(pipeStmts, newPipeCompoundStmt, addBeforeParentLoop, duplicateSet, parentWhileLoop);
      if (insertStmt != null) {
        if (endOfPipeline == null) {
          System.out.println("Did not find end of pipeline");
          System.exit(0);
        } else {
          cStmt.addStatementAfter(endOfPipeline, insertStmt);
          System.out.println("Scalars to duplicate : \n"+duplicateSet.toString());
          insertDuplicatedVariables(duplicateSet, index, insertStmt, addBeforeParentLoop);
          addStatementsBeforeParentLoop(cStmt, addBeforeParentLoop);
        }
      } else
        return 0;

      definedSet.clear();
      duplicateSet.clear();
      addBeforeParentLoop.clear();
    }

    int numStages = index; //pipeStmts.size();
    if (numStages == 0) // there is only one core
      return numStages;
    //Detach statements in the pipeline
    for(Statement currStmt : detachStmts)
      currStmt.detach();
    detachStmts.clear();
    moveToStartOfPipeline.clear();
    tmpStartOfPipeline.clear();
    definedScalars.clear();
    pipeStmts.clear();

    return numStages;

  } // pipelineFCUDAKernels()

  private void pipelineBlockIdxLoop(WhileLoop blockIdxLoop, Procedure proc)
  {
    mIsParentBlockIdxLoop = true;
    int numStages = pipelineFCUDAKernels((CompoundStatement)blockIdxLoop.getBody(), null);
    if (numStages > 0)
      extendLoopExecution(blockIdxLoop, numStages, proc);
  }

  private void handleWhileLoops(LinkedList<WhileLoop> innermostLoops, Procedure proc)
  {
    boolean isHandle = false;
    for (WhileLoop currLoop : innermostLoops) {
      mIsParentBlockIdxLoop = false;
      int numStages = pipelineFCUDAKernels((CompoundStatement)currLoop.getBody(), currLoop);
      if (numStages == 0) {
        System.out.println("Found only one Core in the Loop. Hence proceed to pipeline the next loop.");
        continue;
      }
      System.out.println("Completed pipelineFCUDAKenrels(compoundStatement)");
      extendLoopExecution(currLoop, numStages, proc);
      System.out.println("Completed extendLoopExecution()");
      isHandle = true;
    }
    if (!isHandle) {
      System.out.println("There is no WhileLoop which can be pipelined. Hence proceed to pipeline the BlockIDX loop.");
      pipelineBlockIdxLoop(FCUDAGlobalData.getBlockIdxLoop(proc), proc);
    }
  }

  public void runPass(Procedure proc)
  {
    mProcedure = proc;
    List<FunctionCall> fcudaCores = FCUDAGlobalData.getFcudaCores(mProcedure);
    for (FunctionCall currCore : fcudaCores)
      mCoreStmts.add(FCUDAutils.getClosestParentStmt(currCore));
    int numInnermostLoops = identifyInnerMostWhileLoops(mInnerMostWhileLoops, proc.getBody());

    if (numInnermostLoops == 0)
      pipelineBlockIdxLoop(FCUDAGlobalData.getBlockIdxLoop(proc), proc);
    else
      handleWhileLoops(mInnerMostWhileLoops, proc);
  }

  public void transformProcedure(Procedure proc)
  {
    clear();
    boolean foundCoreInfoPragma = false;
    for (Annotation annot : proc.getAnnotations()) {
      if (annot.get("fcuda") == "coreinfo") {
        foundCoreInfoPragma = true;
        String isPipelined = annot.get("pipeline");
        if (isPipelined == null || isPipelined.equals("no"))
          return;
      } 
    }

    if (!foundCoreInfoPragma)
      return;

    if (FCUDAGlobalData.isConstMemProc(proc)) {
      runPass(FCUDAGlobalData.getKernProcFromStreamProc(proc));
    } else
      runPass(proc);
  }
}
