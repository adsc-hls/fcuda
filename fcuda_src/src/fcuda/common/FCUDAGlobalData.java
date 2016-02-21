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

package fcuda.common;

import java.util.*;
import fcuda.analysis.*;
import fcuda.transforms.*;
import fcuda.utils.*;

import cetus.hir.*;

/**
 *
 * Data structure for maintaining global data related to various
 * FCUDA components. Essentially, provides interfaces for add and 
 * get operations
 *
 */
public class FCUDAGlobalData 
{
  // Keeps track of pragmas we want to add... 
  //private static LinkedList<PragmaAnnotation> mHLSPragmasToAdd;
  private static HashMap<Procedure, LinkedList<PragmaAnnotation>> mHLSPragmasToAdd;
  //private static LinkedList<Procedure> mProceduresNeedingPragmas;
  private static LinkedList<Procedure> mWrapperSingleKern;

  private static int mNumParallelThreadBlocks;

  private static LinkedList<Statement> mFcudaAnnotStmts;
  private static HashMap<FcudaAnnotation, AnnotData> mAnnot2Data;
  private static Iterator mAnnotIter;

  private static Map<Procedure, LinkedList<FunctionCall>> mFcudaCoreList; 
  // Parent
  // Procedure
  // -->
  // list
  // of
  // task
  // calls

  // A TreeMap is needed here (not a HashMap) because it uses a reference to
  // FunctionCall (instead of string copy)
  private static TreeMap<FunctionCall, FcudaCoreData> mFcudaCore2Data;
  private static LinkedList<FunctionCall> mUpdatedFcudaCores;
  private static HashSet<FunctionCall> mFcudaToRemoveOldCores;

  private static HashMap<WhileLoop, WhileLoopData> mWhileLoop2Data;

  // Maps all possible BRAMs to any related data
  private static HashMap<String, BRAMData> mBRAMName2Data;
  // Set of those BRAMs which occur in the arguments of at least one FCUDA
  // task
  private static HashSet<String> mInterfaceBRAMNames;

  // Forloops over block idx
  // *TAN* mInnermostBlockIdxForLoop should be associated with its corresponding proc
  private static HashMap<Procedure, ForLoop> mInnermostBlockIdxForLoop;
  private static HashMap<ForLoop, BlockIdxForLoopData> mBlockIdxForLoop2BlockIdxForLoopData;

  private static HashMap<Procedure, LinkedList<Statement>> mInitializationBlkXStmt;
  private static HashMap<Procedure, Statement> mIncrementBlkXStmt;
  private static HashMap<Procedure, WhileLoop> mBlockIdxLoop;
  private static HashMap<Procedure, WhileLoop> mBlkXLoop;
    private static HashMap<Procedure, IfStatement> mBlkYIf;

  // Statements to ignore during duplication
  private static HashSet<Statement> mIgnoreDuringDuplication;

  private static HashSet<IfStatement> mPipelineCreated;

  private static TreeSet<IDExpression> constMemProcs;

  private static FcudaStreamData streamData;

  private static HashMap<Procedure, DataDepAnalysis> mDepAnalysis;

  private static HashSet<Statement> mThrCondExits; // Hold statements that
  // conditionally exit
  // from eternal while
  // loops

  // Hold variable IDs that have been replicated due to unrolling
  private static HashMap<IDExpression, HashMap<Integer, HashSet<IDExpression>>> mUnrolledIDs;

  private static HashMap<Procedure, Integer> mKernel2TblkDim;

  private static HashMap<Procedure, Procedure> kernProc2StreamProc;

  public static void initialize() 
  {
    mAnnot2Data = new HashMap();
    mFcudaAnnotStmts = new LinkedList();

    mFcudaCoreList = new HashMap<Procedure, LinkedList<FunctionCall>>();
    mFcudaCore2Data = new TreeMap<FunctionCall, FcudaCoreData>();
    mUpdatedFcudaCores = new LinkedList();
    mFcudaToRemoveOldCores = new HashSet();

    mWhileLoop2Data = new HashMap();
    mBRAMName2Data = new HashMap();
    mInterfaceBRAMNames = new HashSet();
    mBlockIdxForLoop2BlockIdxForLoopData = new HashMap();
    mIgnoreDuringDuplication = new HashSet();
    mPipelineCreated = new HashSet();

    mNumParallelThreadBlocks = 1;

    constMemProcs = new TreeSet<IDExpression>();
 
    streamData = null;

    mDepAnalysis = new HashMap<Procedure, DataDepAnalysis>();

    mThrCondExits = new HashSet<Statement>();

    mUnrolledIDs = new HashMap<IDExpression, HashMap<Integer, HashSet<IDExpression>>>();

    mInnermostBlockIdxForLoop = new HashMap<Procedure, ForLoop>();

    mInitializationBlkXStmt = new HashMap<Procedure, LinkedList<Statement>>();
    mIncrementBlkXStmt = new HashMap<Procedure, Statement>();
    mBlockIdxLoop = new HashMap<Procedure, WhileLoop>();
    mBlkXLoop = new HashMap<Procedure, WhileLoop>();
    mBlkYIf = new HashMap<Procedure, IfStatement>();

    mHLSPragmasToAdd = new HashMap<Procedure, LinkedList<PragmaAnnotation>>();

    mWrapperSingleKern = new LinkedList<Procedure>();
    mKernel2TblkDim = new HashMap<Procedure, Integer>();

    kernProc2StreamProc = new HashMap<Procedure, Procedure>();

    FCUDAGlobalData.clear();
  }

  public static void clear() 
  {
    mAnnot2Data.clear();
    mFcudaAnnotStmts.clear();
    mFcudaCoreList.clear();
    mFcudaCore2Data.clear();
    mUpdatedFcudaCores.clear();
    mFcudaToRemoveOldCores.clear();
    mWhileLoop2Data.clear();
    mBRAMName2Data.clear();
    mInterfaceBRAMNames.clear();
    mBlockIdxForLoop2BlockIdxForLoopData.clear();
    mIgnoreDuringDuplication.clear();
    mPipelineCreated.clear();
    constMemProcs.clear();
    mThrCondExits.clear();
    mUnrolledIDs.clear();
    mWrapperSingleKern.clear();
  }

  public static void addUnrolledID(IDExpression proc, int idx, IDExpression id) 
  {
    HashMap<Integer, HashSet<IDExpression>> allIDs;
    HashSet<IDExpression> groupIDs;

    if (!(mUnrolledIDs.keySet().contains(proc))) {
      groupIDs = new HashSet<IDExpression>();
      groupIDs.add(id);
      allIDs = new HashMap<Integer, HashSet<IDExpression>>();
      allIDs.put(idx, groupIDs);

    } else {
      allIDs = mUnrolledIDs.get(proc);
      if (!(allIDs.keySet().contains(idx))) {
        groupIDs = new HashSet<IDExpression>();
        groupIDs.add(id);
        allIDs.put(idx, groupIDs);

      } else {
        allIDs.get(idx).add(id);
      }
    }

    mUnrolledIDs.put(proc, allIDs);

  }

  public static int isUnrolledID(IDExpression proc, IDExpression id) 
  {
    HashMap<Integer, HashSet<IDExpression>> allIDs;
    HashSet<IDExpression> groupIDs;
    int found = -1;

    if (mUnrolledIDs.keySet().contains(proc)) {
      allIDs = mUnrolledIDs.get(proc);
      for (Integer idx : allIDs.keySet()) {
        if (allIDs.get(idx).contains(id)) {
          found = idx;
          break;
        }
      }
    }

    return found;
  }

  public static void printUnrolledIDs() 
  {
    System.out.println("mUnrolledIDs: \n" + mUnrolledIDs.toString());
  }

  public static void addThrCondExit(Statement stmt) 
  {
    mThrCondExits.add(stmt);
  }

  public static boolean isThrCondExit(Statement stmt) 
  {
    boolean contained = false;
    if (mThrCondExits.contains(stmt))
      contained = true;
    return contained;
  }

  public static void recordDataDeps(Procedure proc) 
  {
    DataDepAnalysis dDep = new DataDepAnalysis(proc);
    mDepAnalysis.put(proc, dDep);
    dDep.printVar2Var();
  }

  public static Expression analyzeAccess(Statement transferStmt,
      IDExpression ptrId) 
  {
    Procedure proc = (Procedure) Tools.getAncestorOfType(transferStmt,
        Procedure.class);
    if (proc == null) {
      System.out.println("statement: " + transferStmt.toString()
          + " does not have parent procedure.");
      System.exit(0);
    }

    if (!(mDepAnalysis.keySet().contains(proc))) {
      System.out.println("Procedure " + proc.getName().toString()
          + " has not been data-dep analyzed");
      System.exit(0);
    }

    return mDepAnalysis.get(proc).analyzeAccess(transferStmt, ptrId);
  }

  public static Expression[] getBaseTerms(Procedure proc, String ptrName) 
  {
    if (proc == null) {
      System.out.println("null procedure in parameters");
      System.exit(0);
    }

    if (!(mDepAnalysis.keySet().contains(proc))) {
      System.out.println("Procedure " + proc.getName().toString()
          + " has not been data-dep analyzed (2)");
      System.exit(0);
    }

    return mDepAnalysis.get(proc).getBaseTerms(ptrName);
  }

  public static Set<Expression> getVarDependences(Procedure proc,
      IDExpression id) 
  {
    if (proc == null) {
      System.out.println("null procedure in parameters (2)");
      System.exit(0);
    }

    if (!(mDepAnalysis.keySet().contains(proc))) {
      System.out.println("Procedure " + proc.getName().toString()
          + " has not been data-dep analyzed (3)");
      System.exit(0);
    }

    return mDepAnalysis.get(proc).getVarDeps(id);
  }

  public static void addConstMemProc(Procedure proc, Procedure kernel,
      ForLoop streamLoop, IDExpression streamLoopVar,
      IDExpression streamLoopBound, IDExpression streamLoopUpdate) 
  {
    constMemProcs.add(proc.getName());
    kernProc2StreamProc.put(proc, kernel);

    if (streamData == null)
      streamData = new FcudaStreamData(kernel, streamLoop, streamLoopVar,
          streamLoopBound, streamLoopUpdate);
    else
      streamData.addStream(streamLoop, streamLoopVar, streamLoopBound,
          streamLoopUpdate);
  }

  public static Procedure getKernProcFromStreamProc(Procedure proc)
  {
    if (kernProc2StreamProc.keySet().contains(proc))
        return kernProc2StreamProc.get(proc);
    return null;
  }

  public static boolean isConstMemProc(Procedure proc) 
  {
    return constMemProcs.contains(proc.getName());
  }

  public static Expression getConstMemKern() 
  {
    if (streamData == null)
      return null;
    return streamData.getKernelName();
  }

  public static void addConstMems(Set<IDExpression> mIDs) 
  {
    for (IDExpression mID : mIDs)
      streamData.addConstMemID(mID);
  }

  public static HashSet<IDExpression> getConstMems() {
    HashSet<IDExpression> IDs = new HashSet<IDExpression>();
    IDs.clear();

    if (streamData != null)
      IDs.addAll(streamData.getConstMemIDs());

    return IDs;
  }

  public static boolean isStreamForLoop(ForLoop forStmt) 
  {
    boolean loopMatch = false;
    if (streamData != null) {
      LinkedList<ForLoop> allLoops = streamData.getForLoop();

      if (allLoops.contains(forStmt))
        loopMatch = true;
    }

    return loopMatch;
  }

  public static void addAnnot2Stmts(FcudaAnnotation fc, Statement stmt) 
  {
    checkAndInitAnnotData(fc);
    (mAnnot2Data.get(fc)).addStatement(stmt);
  }

  public static void addAnnot2Body(FcudaAnnotation fc, CompoundStatement stmt) 
  {
    checkAndInitAnnotData(fc);
    (mAnnot2Data.get(fc)).setBody(stmt);
  }

  public static void addAnnot2TaskBody(FcudaAnnotation fc,
      CompoundStatement stmt) 
  {
    checkAndInitAnnotData(fc);
    (mAnnot2Data.get(fc)).setTaskBody(stmt);
  }

  public static void addAnnot2FcudaCore(FcudaAnnotation fc, FunctionCall core) 
  {
    checkAndInitAnnotData(fc);
    (mAnnot2Data.get(fc)).setFcudaCore(core);
  }

  public static CompoundStatement getTaskBody(FcudaAnnotation fc) 
  {
    AnnotData t = checkAndReturnAnnotData(fc);
    return t.getTaskBody();
  }

  public static FunctionCall getFcudaCore(FcudaAnnotation fc) 
  {
    AnnotData t = checkAndReturnAnnotData(fc);
    return t.getFcudaCore();
  }

  public static CompoundStatement getBody(FcudaAnnotation fc) 
  {
    AnnotData t = checkAndReturnAnnotData(fc);
    return t.getBody();
  }

  public static void clearArgsInfo(FcudaAnnotation fc)
  {
    mAnnot2Data.get(fc).clear();
  }

  public static void addArgsInfo(FcudaAnnotation fc,
      Set<IDExpression> taskArgSet, List<IDExpression> taskArgs,
      List<Declaration> taskDecls) 
  {
    checkAndInitAnnotData(fc);
    ((AnnotData) mAnnot2Data.get(fc)).addArgsInfo(taskArgSet, taskArgs,
    taskDecls);
  }

  public static void getArgsInfo(FcudaAnnotation fc,
      Set<IDExpression> taskArgSet, List<IDExpression> taskArgs,
      List<Declaration> taskDecls) 
  {
    AnnotData t = checkAndReturnAnnotData(fc);
    t.getArgsInfo(taskArgSet, taskArgs, taskDecls);
  }

  private static void checkAndInitAnnotData(FcudaAnnotation fc) 
  {
    if (mAnnot2Data.containsKey(fc))
      return;
    else
      mAnnot2Data.put(fc, new AnnotData());
  }

  private static AnnotData checkAndReturnAnnotData(FcudaAnnotation fc) 
  {
    if (mAnnot2Data.containsKey(fc))
      return mAnnot2Data.get(fc);
    else
      Tools.exit("No data found for annotation " + fc.toString());
    return null;
  }

  public static void initAnnotIterator() 
  {
    mAnnotIter = mAnnot2Data.entrySet().iterator();
  }

  public static Map.Entry<FcudaAnnotation, AnnotData> getNextAnnot() 
  {
    if (mAnnotIter.hasNext())
      return (Map.Entry<FcudaAnnotation, AnnotData>) mAnnotIter.next();
    else
      return null;
  }

  public static void addFcudaAnnotationStmt(Statement stmt) 
  {
    mFcudaAnnotStmts.add(stmt);
  }

  public static List<Statement> getFcudaAnnotationStmts() 
  {
    return mFcudaAnnotStmts;
  }

  private static void checkAndInitFcudaCores(Procedure proc) 
  {
    if (!mFcudaCoreList.containsKey(proc))
      mFcudaCoreList.put(proc, new LinkedList<FunctionCall>());
  }

  public static void addFcudaCore(Procedure proc, FunctionCall core) 
  {
    checkAndInitFcudaCores(proc);
    (mFcudaCoreList.get(proc)).add(core);
  }

  public static void setCoreName(FunctionCall core, String name) 
  {
    checkAndInitCoreData(core);
    mFcudaCore2Data.get(core).setName(name);
  }

  public static String getCoreName(FunctionCall core) 
  {
    FcudaCoreData t = checkAndReturnCoreData(core);
    return t.getName();
  }

  public static List<FunctionCall> getFcudaCores(Procedure proc) 
  {
    if (!mFcudaCoreList.containsKey(proc)) {
      System.out.println("Could not find cores for procedure "
          + proc.getName().toString());
      System.exit(0);
    }
    return mFcudaCoreList.get(proc);
  }

  public static List<FunctionCall> getAllFcudaCores() 
  {
    List<FunctionCall> allCores = new LinkedList<FunctionCall>();
    for (Procedure proc : mFcudaCoreList.keySet())
      allCores.addAll(mFcudaCoreList.get(proc));
    return allCores;
  }

  public static void addEnableSignal(FunctionCall core, IDExpression enable) 
  {
    checkAndInitCoreData(core);
    mFcudaCore2Data.get(core).setEnableSignal(enable);
  }

  public static IDExpression getEnableSignal(FunctionCall core) 
  {
    FcudaCoreData t = checkAndReturnCoreData(core);
    return t.getEnableSignal();
  }

  public static void setCoreType(FunctionCall core, int type) 
  {
    checkAndInitCoreData(core);
    mFcudaCore2Data.get(core).setType(type);
  }

  public static int getCoreType(FunctionCall core) 
  {
    FcudaCoreData t = checkAndReturnCoreData(core);
    return t.getType();
  }

  public static void setNumCores(FunctionCall core, int num) 
  {
    checkAndInitCoreData(core);
    mFcudaCore2Data.get(core).setNumCores(num);
  }

  public static int getNumCores(FunctionCall core) 
  {
    FcudaCoreData t = checkAndReturnCoreData(core);
    return t.getNumCores();
  }

  public static void addCommonArgsIndex(FunctionCall core,
      LinkedList<Integer> commonArgsIndex) 
  {
    checkAndInitCoreData(core);
    mFcudaCore2Data.get(core).addCommonArgsIndex(commonArgsIndex);
  }

  public static LinkedList<Integer> getCommonArgsIndex(FunctionCall core) {
    FcudaCoreData t = checkAndReturnCoreData(core);
    return t.getCommonArgsIndex();
  }

  // *AP* Update indexes after position <position> by <update>
  public static void updateCommonArgsIndex(FunctionCall core, int position,
      int update) 
  {
    checkAndInitCoreData(core);
    mFcudaCore2Data.get(core).updateCommonArgsIndex(position, update);
  }

  private static void checkAndInitCoreData(FunctionCall core) 
  {
    if (!mFcudaCore2Data.containsKey(core)) {
      mFcudaCore2Data.put(core, new FcudaCoreData());
      System.out.println("Creating new FcudaCoreData for core: "
          + core.toString());
    }
  }

  private static FcudaCoreData checkAndReturnCoreData(FunctionCall core) 
  {
    if (!mFcudaCore2Data.containsKey(core)) {
      System.out.println("Could not find core data for statement "
          + core.toString());
      System.exit(0);
    }
    return mFcudaCore2Data.get(core);
  }

  public static void setWhileLoopCondition(WhileLoop stmt, Expression cond) 
  {
    checkAndInitWhileLoopData(stmt);
    mWhileLoop2Data.get(stmt).setWhileLoopCondition(cond);
  }

  public static Expression getWhileLoopCondition(WhileLoop stmt) 
  {
    WhileLoopData t = checkAndReturnWhileLoopData(stmt);
    return t.getWhileLoopCondition();
  }

  public static void setGuardInitStmt(WhileLoop loop, Statement initStmt) 
  {
    checkAndInitWhileLoopData(loop);
    mWhileLoop2Data.get(loop).setGuardInitStmt(initStmt);
  }

  public static Statement getGuardInitStmt(WhileLoop loop) 
  {
    WhileLoopData t = checkAndReturnWhileLoopData(loop);
    return t.getGuardInitStmt();
  }

  public static void setLoopVarInitStmt(WhileLoop loop, Statement initVarStmt) 
  {
    checkAndInitWhileLoopData(loop);
    mWhileLoop2Data.get(loop).setInitLoopVar(initVarStmt);
  }

  public static Statement getLoopVarInitStmt(WhileLoop loop) {
    WhileLoopData t = checkAndReturnWhileLoopData(loop);
    return t.getInitLoopVar();
  }

  public static void addWhileLoopData(WhileLoop loop, IDExpression guard,
      IfStatement guardIf) 
  {
    checkAndInitWhileLoopData(loop);
    WhileLoopData t = mWhileLoop2Data.get(loop);
    t.setGuardVar(guard);
    t.setGuardIf(guardIf);
  }

  public static IfStatement getGuardIf(WhileLoop loop) 
  {
    WhileLoopData t = checkAndReturnWhileLoopData(loop);
    return t.getGuardIf();
  }

  public static IDExpression getGuardVar(WhileLoop loop) 
  {
    WhileLoopData t = checkAndReturnWhileLoopData(loop);
    return t.getGuardVar();
  }

  public static WhileLoopData checkAndReturnWhileLoopData(WhileLoop stmt) 
  {
    if (!mWhileLoop2Data.containsKey(stmt)) {
      System.out.println("Could not find while loop in map " + stmt.toString());
      System.exit(0);
    }
    return mWhileLoop2Data.get(stmt);
  }

  private static void checkAndInitWhileLoopData(WhileLoop stmt) 
  {
    if (!mWhileLoop2Data.containsKey(stmt))
      mWhileLoop2Data.put(stmt, new WhileLoopData());
  }

  public static Expression getEnableExpression(FunctionCall core) 
  {
    return core.getArgument(0);
  }

  // *TAN* This will never work as it conflicts with the FcudaCoreData
  // 		 unless we change the type of enableSignal from IDExpression
  //		 to Expression in FcudaCoreData.
  public static void setEnableExpression(FunctionCall core, Expression expr) 
  {
    List<Expression> old_args = core.getArguments();
    LinkedList<Expression> new_args = new LinkedList<Expression>();	
    old_args.set(0, expr);
    for (Expression arg : old_args)
    {
      new_args.add(arg.clone());
    }
    core.setArguments(new_args);
    checkAndInitCoreData(core);
  }

  public static boolean isCreatedWhileLoop(WhileLoop t) 
  {
    return mWhileLoop2Data.containsKey(t);
  }

  public static int getNumNonIDArguments() 
  {
    return 4;
  }

  public static IDExpression getNonIDArgument(FunctionCall core, int blockId,
      int paramId) 
  {
    String varId = "";
    switch (paramId) {
      case 0:
        varId += "enableSignal_" + FCUDAGlobalData.getCoreName(core)
          + DuplicateForFCUDA.getPostFix(blockId);
        break;
      case 1:
        varId += "blockDim";
        break;
      case 2:
        varId += "gridDim";
        break;
      case 3:
        varId += "blockIdx_" + FCUDAGlobalData.getCoreName(core)
          + DuplicateForFCUDA.getPostFix(blockId);
        break;
      default:
        System.out.println("Only 4 FCUDA non ID arguments exist(1)");
        System.exit(0);
    }
    return (new NameID(varId));
  }

  public static Declaration getNonIDParameter(FunctionCall core, int blockId,
      int paramId) 
  {
    Specifier type = null;
    switch (paramId) {
      case 0:
        type = Specifier.INT;
        break;
      case 1:
      case 2:
      case 3:
         type = new UserSpecifier(new NameID("dim3"));
        break;
      default:
        System.out.println("Only 4 FCUDA non ID arguments exist(2)");
        System.exit(0);
    }
    return new VariableDeclaration(type, new VariableDeclarator(
          FCUDAGlobalData.getNonIDArgument(core, blockId, paramId)));
  }

  public static void checkAndInitBRAMData(String name) 
  {
    if (!mBRAMName2Data.containsKey(name)) {
      BRAMData t = new BRAMData();
      mBRAMName2Data.put(name, t);
    }
  }

  public static BRAMData checkAndReturnBRAMData(String name) 
  {
    if (!mBRAMName2Data.containsKey(name)) {
      System.out.println("Does not contain BRAM named " + name);
      System.exit(0);
    }
    return mBRAMName2Data.get(name);
  }

  public static boolean isBRAM(IDExpression id) 
  {
    return mBRAMName2Data.containsKey(id.toString());
  }

  public static void setBRAMSet(HashMap<IDExpression, Integer> bramSet) 
  {
    for (IDExpression bramId : bramSet.keySet()) {
      checkAndInitBRAMData(bramId.toString());
      ((BRAMData) mBRAMName2Data.get(bramId.toString())).setDim(bramSet
      .get(bramId));
    }
  }

  public static void addInterfaceBRAM(IDExpression bramId) 
  {
    mInterfaceBRAMNames.add(bramId.toString());
  }

  public static void addInterfaceBRAM(IDExpression bramId,
      FunctionCall coreCall) 
  {
    addInterfaceBRAM(bramId);
    checkAndInitBRAMData(bramId.toString());
    mBRAMName2Data.get(bramId.toString()).addFCUDACore(coreCall);
  }

  public static boolean isInterfaceBRAM(IDExpression id) 
  {
    return mInterfaceBRAMNames.contains(id.toString());
  }

  public static HashSet<FunctionCall> getFCUDACoresForBRAM(IDExpression id) 
  {
    BRAMData t = checkAndReturnBRAMData(id.toString());
    return t.getFCUDACores();
  }

  public static void partitionBRAM(IDExpression origBRAM,
      IDExpression partBRAM) 
  {
    BRAMData t = checkAndReturnBRAMData(origBRAM.toString());
    mBRAMName2Data.put(partBRAM.toString(), t);

    if (isInterfaceBRAM(origBRAM))
      addInterfaceBRAM(partBRAM);
  }

  public static void removeBRAM(IDExpression id) 
  {
    if (mBRAMName2Data.containsKey(id.toString()))
      mBRAMName2Data.remove(id.toString());

    if (isInterfaceBRAM(id))
      mInterfaceBRAMNames.remove(id.toString());
  }

  public static boolean isCreatedForLoop(ForLoop forLoop) 
  {
    return mBlockIdxForLoop2BlockIdxForLoopData.containsKey(forLoop);
  }

  public static void setInitializationBlkXStmt(Procedure proc, Statement stmt) 
  {
    if (!mInitializationBlkXStmt.containsKey(proc))
      mInitializationBlkXStmt.put(proc, new LinkedList<Statement>());
    mInitializationBlkXStmt.get(proc).add(stmt);
  }

  public static LinkedList<Statement> getInitializationBlkXStmt(Procedure proc) 
  {
    if (mInitializationBlkXStmt.containsKey(proc))
      return mInitializationBlkXStmt.get(proc);
    return null;
  }

  public static void setIncrementBlkXStmt(Procedure proc, Statement stmt) 
  {
    mIncrementBlkXStmt.put(proc, stmt);
  }

  public static Statement getIncrementBlkXStmt(Procedure proc) 
  {
    if (mIncrementBlkXStmt.containsKey(proc))
      return mIncrementBlkXStmt.get(proc);
    return null;
  }

  public static void setBlockIdxLoop(Procedure proc, WhileLoop loop) 
  {
    mBlockIdxLoop.put(proc, loop);
  }

  public static WhileLoop getBlockIdxLoop(Procedure proc) 
  {
    if (mBlockIdxLoop.containsKey(proc))
      return mBlockIdxLoop.get(proc);
    return null;
  }

  public static void setBlkXLoop(Procedure proc, WhileLoop loop) 
  {
    mBlkXLoop.put(proc, loop);
  }

  public static WhileLoop getBlkXLoop(Procedure proc) 
  {
    if (mBlkXLoop.containsKey(proc))
      return mBlkXLoop.get(proc);
    return null;
  }

  public static void setBlkYIf(Procedure proc, IfStatement ifStmtBlkY) 
  {
    mBlkYIf.put(proc, ifStmtBlkY);
  }

  public static IfStatement getBlkYIf(Procedure proc) 
  {
    if (mBlkYIf.containsKey(proc))
      return mBlkYIf.get(proc);
    return null;
  }

  public static void setInnermostBlockIdxForLoop(Procedure proc, ForLoop loop) 
  {
    mInnermostBlockIdxForLoop.put(proc, loop);
    checkAndInitBlockIdxForLoopData(loop);
  }

  public static ForLoop getInnermostBlockIdxForLoop(Procedure proc) 
  {
    if (mInnermostBlockIdxForLoop.containsKey(proc))
      return mInnermostBlockIdxForLoop.get(proc);
    return null;
  }

  public static void setBlockIdxForLoopAssignmentStmt(ForLoop loop,
      ExpressionStatement stmt) 
  {
    checkAndInitBlockIdxForLoopData(loop);
    (mBlockIdxForLoop2BlockIdxForLoopData.get(loop))
      .setAssignmentStmt(stmt);
  }

  public static ExpressionStatement getBlockIdxForLoopAssignmentStmt(
      ForLoop loop) 
  {
    return (checkAndReturnBlockIdxForLoopData(loop)).getAssignmentStmt();
  }

  private static void checkAndInitBlockIdxForLoopData(ForLoop loop) 
  {
    if (!mBlockIdxForLoop2BlockIdxForLoopData.containsKey(loop))
      mBlockIdxForLoop2BlockIdxForLoopData.put(loop,
        new BlockIdxForLoopData());
  }

  private static BlockIdxForLoopData checkAndReturnBlockIdxForLoopData(
      ForLoop loop) 
  {
    if (!mBlockIdxForLoop2BlockIdxForLoopData.containsKey(loop)) {
      System.out.println("No for loop data found for " + loop.toString());
      System.exit(0);
    }
    BlockIdxForLoopData t = mBlockIdxForLoop2BlockIdxForLoopData.get(loop);
    return t;
  }

  public static void addIgnoreDuringDuplication(Statement stmt) 
  {
    mIgnoreDuringDuplication.add(stmt);
  }

  public static HashSet<Statement> getIgnoreDuringDuplicationSet() 
  {
    return mIgnoreDuringDuplication;
  }

  public static void addPipelineCreated(IfStatement stmt) 
  {
    mPipelineCreated.add(stmt);
  }

  public static boolean isPipelineCreated(IfStatement stmt) 
  {
    return mPipelineCreated.contains(stmt);
  }

  public static void copyFcudaCoreData(FunctionCall refCall,
      FunctionCall newCall) 
  {
    mUpdatedFcudaCores.add(newCall);
    mFcudaToRemoveOldCores.add(refCall);

    FcudaCoreData t = checkAndReturnCoreData(refCall);
    FcudaCoreData newData = new FcudaCoreData();
    newData.createCopy(t);
    mFcudaCore2Data.put(newCall, newData);
  }

  public static void prepareToResetFCUDACores() 
  {
    mUpdatedFcudaCores.clear();
    mFcudaToRemoveOldCores.clear();
  }

  public static void updateFCUDACores(Procedure proc) 
  {
    LinkedList<FunctionCall> tmpList = new LinkedList();
    for (FunctionCall currCall : mFcudaToRemoveOldCores)
      mFcudaCore2Data.remove(currCall);

    if (!mFcudaCoreList.containsKey(proc)) {
      System.out.println("No cores for procedure " + proc.getName().toString());
      System.exit(0);
    }
    for (FunctionCall currCall : mFcudaCoreList.get(proc))
      if (!mFcudaToRemoveOldCores.contains(currCall))
        tmpList.add(currCall);

    mFcudaCoreList.get(proc).clear();
    mFcudaCoreList.get(proc).addAll(tmpList);
    mFcudaCoreList.get(proc).addAll(mUpdatedFcudaCores);

    mUpdatedFcudaCores.clear();
    mFcudaToRemoveOldCores.clear();
    tmpList.clear();
  }

  public static Set getCoreNames() 
  {
    return mFcudaCore2Data.keySet();
  }

  public static void setNumParallelThreadBlocks(Procedure proc, int arg) 
  {
    // mNumParallelThreadBlocks = a;
    // *** Eric*** change this function according to get function.

    FunctionCall core = mFcudaCoreList.get(proc).getFirst();
    FcudaCoreData cData = checkAndReturnCoreData(core);
    cData.setNumCores(arg);
  }

  public static int getNumParallelThreadBlocks(Procedure proc) 
  {
    if (!mFcudaCoreList.containsKey(proc)) {
      System.out.println("No cores found for procedure "
          + proc.getName().toString());
      System.exit(0);
    }

    // **AP** Assuming that all tasks have same unrolling
    // **AP** if assumption incorrect should change this function
    FunctionCall core = mFcudaCoreList.get(proc).getFirst();
    FcudaCoreData cData = checkAndReturnCoreData(core);
    System.out.println("cData getNumCores" + proc.toString() + cData.getNumCores());
    return cData.getNumCores();

  }

  public static void addHLSPragma(Procedure proc, PragmaAnnotation p) 
  {
    if (!mHLSPragmasToAdd.containsKey(proc)) {
      LinkedList<PragmaAnnotation> listPragmas = new LinkedList<PragmaAnnotation>();
      mHLSPragmasToAdd.put(proc, listPragmas);
    }
    mHLSPragmasToAdd.get(proc).add(p);
  }

  public static void addListHLSPragmas(Procedure proc, LinkedList<PragmaAnnotation> p) 
  {
    mHLSPragmasToAdd.put(proc, p);
  }

  public static List<PragmaAnnotation> getHLSPragmas(Procedure proc) 
  {
    return mHLSPragmasToAdd.get(proc);
  }

  public static List<Procedure> getListWrapperSingleKern() 
  {
    return mWrapperSingleKern;
  }

  public static void addWrapperSingleKern(Procedure p) 
  {
    mWrapperSingleKern.add(p);
  }

  public static void setKernTblkDim(Procedure proc, int numDims)
  {
    mKernel2TblkDim.put(proc, numDims);
  }

  public static int getKernTblkDim(Procedure proc)
  {
    if (mKernel2TblkDim.keySet().contains(proc))
      return mKernel2TblkDim.get(proc);
    return 0;
  }
}
