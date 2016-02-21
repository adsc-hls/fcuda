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

package fcuda.utils;
import fcuda.common.*;
import fcuda.transforms.*;
import java.util.*;
import cetus.hir.*;


/* Utility functions and structures for FCUDA */


public class FCUDAutils
{

  protected static HashMap<String, List<Procedure>> taskMap;
  protected static HashMap<String, Set<Expression>> thrDefMap; // ThreadIdx based definitions
  protected static HashMap<IDExpression, TreeSet<Expression>> thrDefIDs;    // IDExpressions that are directly defined as expresions of threadIdx
  protected static HashMap<IDExpression, TreeSet<Expression>> thrDepIDs;    // IDExpressions that are indirectly dependent on threadIdx

  protected static HashMap<Procedure, String> mTask2Unroll;
  protected static HashMap<Procedure, String> mTask2MemPart;
  protected static HashMap<Procedure, List<String>> mTask2SplitArray;
  protected static HashMap<Procedure, List<String>> mTask2Shape;
  protected static HashMap<Procedure, String> mTask2Type;

  //Eric
  protected static HashMap<Procedure, List<String>> mTask2SharedArray;
  protected static HashMap<Procedure, List<String>> mTask2NonSharedArray;
  protected static HashMap<Procedure, String> mTask2TotalBram;
  protected static HashMap<Procedure, String> mTask2CoreBram;
  protected static HashMap<Procedure, String> mTask2SharedBram;
  protected static HashMap<Procedure, String> mTask2TotalBlock;

  static {
    taskMap = new HashMap<String, List<Procedure>>();
    thrDefMap = new HashMap<String, Set<Expression>>();
    thrDefIDs = new HashMap<IDExpression, TreeSet<Expression>>();
    thrDepIDs = new HashMap<IDExpression, TreeSet<Expression>>();
    mTask2Unroll = new HashMap<Procedure, String>();
    mTask2MemPart = new HashMap<Procedure, String>();
    mTask2SplitArray = new HashMap<Procedure, List<String>>();
    mTask2Shape = new HashMap<Procedure, List<String>>();
    mTask2Type = new HashMap<Procedure, String>();

    //Eric
    mTask2SharedArray    = new HashMap<Procedure, List<String>>();
    mTask2NonSharedArray = new HashMap<Procedure, List<String>>();
    mTask2TotalBram      = new HashMap<Procedure, String>();
    mTask2CoreBram       = new HashMap<Procedure, String>();
    mTask2SharedBram     = new HashMap<Procedure, String>();
    mTask2TotalBlock     = new HashMap<Procedure, String>();
  }



  public static void addTaskMapping(String kern, Procedure task)
  {
    if(taskMap.containsKey(kern))
      (taskMap.get(kern)).add(task);
    else {
      List<Procedure> taskLst = new LinkedList<Procedure>();
      taskLst.add(task);
      taskMap.put(kern, taskLst);
    }
  }

  public static HashMap<String, List<Procedure>> getTaskMap()
  {
    return taskMap;
  }

  public static void setTaskUnroll(Procedure task, String unrol)
  {
    if(mTask2Unroll.containsKey(task))
      mTask2Unroll.remove(task);

    mTask2Unroll.put(task, unrol);
  }

  public static void setTaskMempart(Procedure task, String mpart)
  {
    if(mTask2MemPart.containsKey(task))
      mTask2MemPart.remove(task);

    mTask2MemPart.put(task, mpart);
  }


  public static void setTaskSplitArray(Procedure task, List<String> arrList)
  {
    if(mTask2SplitArray.containsKey(task))
      mTask2SplitArray.remove(task);

    mTask2SplitArray.put(task, arrList);

  }

  public static void setTaskShape(Procedure task, List<String> shapeList)
  {
    if (mTask2Shape.containsKey(task))
      mTask2Shape.remove(task);

    mTask2Shape.put(task, shapeList);
  }

  //Eric
  public static void setTaskNonSharedArray(Procedure task, List<String> arrList)
  {
    if(mTask2NonSharedArray.containsKey(task))
      mTask2NonSharedArray.remove(task);

    mTask2NonSharedArray.put(task, arrList);

  }

  public static void setTaskSharedArray(Procedure task, List<String> arrList)
  {
    if(mTask2SharedArray.containsKey(task))
      mTask2SharedArray.remove(task);

    mTask2SharedArray.put(task, arrList);

  }

  public static void setTaskBramCore(Procedure task, List<String> arrList)
  {
    if(mTask2TotalBram.containsKey(task))
      mTask2TotalBram.remove(task);
    mTask2TotalBram.put(task,  arrList.get(0));

    if(mTask2CoreBram.containsKey(task))
      mTask2CoreBram.remove(task);
    mTask2CoreBram.put(task,   arrList.get(1));

    if(mTask2SharedBram.containsKey(task))
      mTask2SharedBram.remove(task);
    mTask2SharedBram.put(task, arrList.get(2));

    if(mTask2TotalBlock.containsKey(task))
      mTask2TotalBlock.remove(task);
    mTask2TotalBlock.put(task, arrList.get(3));

  }

  public static void setTaskType(Procedure task, String tp)
  {
    if(mTask2Type.containsKey(task))
      mTask2Type.remove(task);

    mTask2Type.put(task, tp);

  }

  public static List<Procedure> getTaskMapping(String kern)
  {
    List<Procedure> kernTasks = null;

    if(taskMap.containsKey(kern))
      kernTasks = taskMap.get(kern);

    return kernTasks;
  }

  public static int getTaskUnroll(Procedure task)
  {
    int unroll = 1;

    if(mTask2Unroll.containsKey(task))
      unroll = Integer.parseInt(mTask2Unroll.get(task));

    return unroll;
  }


  public static int getTaskMempart(Procedure task)
  {
    int mpart = 1;

    if(mTask2MemPart.containsKey(task))
      mpart = Integer.parseInt(mTask2MemPart.get(task));

    return mpart;
  }

  public static List<String> getTaskSplitArray(Procedure task)
  {
    List<String> arrays = null;

    if(mTask2SplitArray.containsKey(task))
      arrays = mTask2SplitArray.get(task);

    return arrays;
  }

  public static List<String> getTaskShape(Procedure task)
  {
    List<String> shapes = null;
    if (mTask2Shape.containsKey(task))
      shapes = mTask2Shape.get(task);

    return shapes;
  }

  //Eric
  public static List<String> getTaskNonSharedArray(Procedure task)
  {
    List<String> arrays = null;

    if(mTask2NonSharedArray.containsKey(task))
      arrays = mTask2NonSharedArray.get(task);

    return arrays;
  }

  public static List<String> getTaskSharedArray(Procedure task)
  {
    List<String> arrays = null;

    if(mTask2SharedArray.containsKey(task))
      arrays = mTask2SharedArray.get(task);

    return arrays;
  }


  public static int getTaskTotalBram(Procedure task)
  {
    int totalBram = 1;
    if(mTask2TotalBram.containsKey(task))
      totalBram = Integer.parseInt(mTask2TotalBram.get(task));

    return totalBram;
  }

  public static int getTaskCoreBram(Procedure task)
  {
    int coreBram = 1;
    if(mTask2CoreBram.containsKey(task))
      coreBram = Integer.parseInt(mTask2CoreBram.get(task));
    return coreBram;
  }

  public static int getTaskSharedBram(Procedure task)
  {
    int sharedBram = 1;
    if(mTask2SharedBram.containsKey(task))
      sharedBram = Integer.parseInt(mTask2SharedBram.get(task));
    return sharedBram;
  }

  public static int getTaskTotalBlock(Procedure task)
  {
    int totalBlock = 1;
    if(mTask2TotalBlock.containsKey(task))
      totalBlock = Integer.parseInt(mTask2TotalBlock.get(task));
    return totalBlock;
  }

  public static String getTaskType(Procedure task)
  {
    String tp = null;

    if(mTask2Type.containsKey(task))
      tp = mTask2Type.get(task);

    return tp;
  }

  public static void addThrDefMapping(String kern, Expression thrDef)
  {
    if(thrDefMap.containsKey(kern))
      (thrDefMap.get(kern)).add(thrDef);
    else {
      Set<Expression> thrDefSet = new TreeSet<Expression>();
      thrDefSet.add(thrDef);
      thrDefMap.put(kern, thrDefSet);
    }
  }

  public static Set<Expression> getThrDefMapping(String kern)
  {
    Set<Expression> thrDefSet = null;

    if(thrDefMap.containsKey(kern))
      thrDefSet = thrDefMap.get(kern);

    return thrDefSet;
  }

  public static void addThrDefID(IDExpression id, Expression thrDef)
  {
    if(thrDefIDs.containsKey(id))
      (thrDefIDs.get(id)).add(thrDef);
    else {
      TreeSet<Expression> thrDefSet = new TreeSet<Expression>();
      thrDefSet.add(thrDef);
      thrDefIDs.put(id, thrDefSet);
    }
  }

  public static Expression getThrDefExp(IDExpression id)
  {
    TreeSet<Expression> thrDefSet = null;
    Expression thrDefExp = null;

    if(thrDefIDs.containsKey(id)) {
      thrDefSet = thrDefIDs.get(id);
      if(thrDefSet.size() > 1) {
        for (Expression exp : thrDefSet)
          System.out.println(" Def: "+exp.toString());
        System.out.println(id.toString()+" has more than one ThreadIdx definitions: Not handled yet!");
        System.exit(0);
      }
      thrDefExp = (Expression) thrDefSet.first();
    }

    return thrDefExp;
  }

  public static void addThrDepID(IDExpression id, Expression thrDep)
  {
    if(thrDepIDs.containsKey(id))
      (thrDepIDs.get(id)).add(thrDep);
    else {
      TreeSet<Expression> thrDepSet = new TreeSet<Expression>();
      thrDepSet.add(thrDep);
      thrDepIDs.put(id, thrDepSet);
    }
  }

  public static Expression getThrDepExp(IDExpression id)
  {
    TreeSet<Expression> thrDepSet = null;
    Expression thrDepExp = null;

    if(thrDepIDs.containsKey(id)) {
      thrDepSet = thrDepIDs.get(id);
      if(thrDepSet.size() > 1) {
        for (Expression exp : thrDepSet)
          System.out.println(" Dep: "+exp.toString());
        System.out.println(id.toString()+" has more than one ThreadIdx dependent definitions: not handled yet!");
        System.exit(0);
      }
      thrDepExp = (Expression) thrDepSet.first();
    }

    return thrDepExp;
  }

  public static SymbolTable getSymbolTableUnchecked(Traversable input)
  {
    Traversable t = input;
    SymbolTable st = null;
    while (t != null && st == null)
    {
      try {
        st = (SymbolTable)t;
      } catch (ClassCastException e) {
        t = t.getParent();
        st = null;
        continue;
      }
      return st;
    }
    return null;
  }

  public static SymbolTable getSymbolTable(Traversable input)
  {
    SymbolTable st = FCUDAutils.getSymbolTableUnchecked(input);
    if(st == null) {
      System.out.println("Could not find symbol table for traversable "+input.toString());
      System.exit(0);
    }
    return st;
  }

  public static VariableDeclarator getDeclaratorFromDeclaration(VariableDeclaration varD, IDExpression idExpr)
  {
    VariableDeclarator vDecl = FCUDAutils.getDeclaratorFromDeclarationUnchecked(varD, idExpr);
    if (vDecl == null) {
      System.out.println(" Could not find variable declarator for "+idExpr.toString()+" in declaration :"+varD.toString());
      System.exit(0);
    }
    return vDecl;
  }

  public static VariableDeclarator getDeclaratorFromDeclarationUnchecked(VariableDeclaration varD, IDExpression idExpr)
  {
    int i;
    for (i=0;i<varD.getNumDeclarators();++i) {
      Declarator decl = varD.getDeclarator(i);
      if(decl == null || !(decl instanceof VariableDeclarator))
        return null;
      if (decl.getID().equals(idExpr))
        return (VariableDeclarator)decl;
    }
    return null;
  }

  public static VariableDeclarator getVariableDeclarator(IDExpression idExpr)
  {
    VariableDeclarator vDecl = FCUDAutils.getVariableDeclaratorUnchecked(idExpr);
    if (vDecl == null) {
      System.out.println("No variable declarator found for "+idExpr.toString());
      System.exit(0);
    }
    return vDecl;
  }

  public static VariableDeclarator getVariableDeclaratorUnchecked(IDExpression idExpr)
  {
    VariableDeclaration varDecl = FCUDAutils.getVariableDeclarationUnchecked(idExpr);
    if (varDecl == null)
      return null;
    return FCUDAutils.getDeclaratorFromDeclarationUnchecked(varDecl, idExpr);
  }

  public static VariableDeclaration getVariableDeclaration(IDExpression idExpr)
  {
    VariableDeclaration tmp = getVariableDeclarationUnchecked(idExpr);
    if (tmp == null) {
      System.out.println("No variable declaration found for "+idExpr.toString());
      System.exit(0);
    }
    return tmp;
  }

  public static VariableDeclaration getVariableDeclarationUnchecked(IDExpression idExpr)
  {
    // Find nearest symbol table parent (st)
    SymbolTable st = FCUDAutils.getSymbolTable(idExpr);
    if (st == null)
      return null;
    return FCUDAutils.getVariableDeclarationUnchecked(idExpr, st);
  }

  public static VariableDeclaration getVariableDeclarationUnchecked(IDExpression idExpr, SymbolTable st)
  {
    if (st == null)
      return null;
    Declaration decl = Tools.findSymbol(st, idExpr);
    if (decl == null)
      return null;
    if (!(decl instanceof VariableDeclaration))
      return null;
    VariableDeclaration varDecl = (VariableDeclaration)decl;
    return varDecl;
  }

  public static Statement getClosestParentStmt(Traversable inputE)
  {
    Traversable t = inputE;
    while (t != null && !(t instanceof Statement))
      t = t.getParent();
    if (t == null) {
      System.out.println(" Could not find closest parent statement for "+inputE.toString());
      System.exit(0);
    }
    return (Statement)t;
  }


  public static VariableDeclaration createSharedArrayDeclaration(VariableDeclaration currDecl, VariableDeclarator currDeclor,
      List<Expression> dimSpecs)
  {
    LinkedList<Specifier> leadSpecs = new LinkedList<Specifier>();
    LinkedList<Specifier> trailSpecs = new LinkedList<Specifier>();

    IDExpression idExpr = new NameID(currDeclor.getSymbolName()+"_block");
    leadSpecs.addAll(currDeclor.getSpecifiers());
    trailSpecs.addAll(currDeclor.getTrailingSpecifiers());

    ArraySpecifier aSpec = new ArraySpecifier();
    aSpec.setDimensions(dimSpecs);
    trailSpecs.add(aSpec);
    VariableDeclarator arrayDeclor = new VariableDeclarator(leadSpecs, idExpr, trailSpecs);

    leadSpecs.clear();
    trailSpecs.clear();
    leadSpecs.addAll(currDecl.getSpecifiers());
    leadSpecs.add(0, Specifier.SHARED);
    VariableDeclaration arrayDecl = new VariableDeclaration(leadSpecs, arrayDeclor);

    leadSpecs.clear();
    trailSpecs.clear();
    return arrayDecl;
  }

  public static VariableDeclaration createSharedArrayDeclaration(VariableDeclaration currDecl, VariableDeclarator currDeclor,
      int numDims)
  {
    LinkedList<Expression> dimList = new LinkedList<Expression>();
    for (int i=0;i<numDims;++i)
      dimList.add((Expression)null);
    return createSharedArrayDeclaration(currDecl, currDeclor, dimList);
  }

  public static Expression getThreadIdxArrayAccess(VariableDeclaration arrayDecl, int numDims)
  {
    Expression tx = MCUDAUtils.Tidx.getId(0);
    Expression ty = MCUDAUtils.Tidx.getId(1);
    Expression tz = MCUDAUtils.Tidx.getId(1);

    LinkedList<Expression> arrayIdx = new LinkedList<Expression>();
    if (numDims == 3)
      arrayIdx.add(tz);
    if (numDims > 1)
      arrayIdx.add(ty);
    arrayIdx.add(tx);

    IDExpression arrayId = new Identifier((Symbol)arrayDecl.getDeclarator(0));

    ArrayAccess aExpr = new ArrayAccess(arrayId, arrayIdx);
    arrayIdx.clear();
    return aExpr;
  }

  public static boolean isCudaDim3Struct(IDExpression structId, String cmpName)
  {
    String name = structId.toString();
    if (cmpName.length() == 0)
      return (name.equals("blockDim") || name.equals("threadIdx") || name.equals("blockIdx") || name.equals("gridDim"));
    else
      return (name.equals(cmpName));
  }

  public static boolean isCudaDim3Struct(Expression expr, String cmpName)
  {
    if (expr instanceof IDExpression)
      return FCUDAutils.isCudaDim3Struct((IDExpression)expr, cmpName);
    if (expr instanceof AccessExpression)
    {
      AccessExpression aExpr = (AccessExpression)expr;
      if (aExpr.getLHS() instanceof IDExpression)
        return 	FCUDAutils.isCudaDim3Struct((IDExpression)aExpr.getLHS(), cmpName);
    }
    return false;
  }

  public static boolean isCudaDim3Struct(Expression expr)
  {
    return FCUDAutils.isCudaDim3Struct(expr,"");
  }

  //Something which is not array or pointer - does not handle structures etc.
  public static boolean isScalar(VariableDeclarator varDecl)
  {
    if(Tools.isArray(varDecl) || Tools.isPointer(varDecl))
      return false;
    return true;
  }

  //Something which is not array or pointer - does not handle structures etc.
  public static boolean isArrayOrPointer(IDExpression idExpr)
  {
    VariableDeclarator varDecl = FCUDAutils.getVariableDeclarator(idExpr);
    return (Tools.isArray(varDecl) || Tools.isPointer(varDecl));
  }

  public static boolean isThreadIdx(IDExpression idExpr)
  {
    return (idExpr.equals(MCUDAUtils.Tidx.getId().get(0)));
  }


  public static List<Expression> getTerms(Expression e)
  {
    LinkedList<Expression> retList = new LinkedList<Expression>();
    FCUDAutils.getTerms(e ,retList);
    return retList;
  }

  public static void getTerms(Expression e, List<Expression> retList)
  {
    if (e instanceof BinaryExpression && ((BinaryExpression)e).getOperator() == BinaryOperator.ADD)
    {
      getTerms(((BinaryExpression)e).getLHS(), retList);
      getTerms(((BinaryExpression)e).getRHS(), retList);
    }
    else
      retList.add(e);
  }


  public static boolean isSharedMem(Expression access)
  {
    IDExpression arrayId = null;
    if (access instanceof ArrayAccess)
      arrayId = (IDExpression)((ArrayAccess)access).getArrayName();
    if (access instanceof IDExpression)
      arrayId = (IDExpression)access;

    VariableDeclaration varDecl = FCUDAutils.getVariableDeclaration(arrayId);
    List<Specifier> specList = varDecl.getSpecifiers();
    for (Specifier currSpec : specList) {
      if (currSpec == Specifier.SHARED)
        return true;
    }
    return false;
  }

  public static void addAfterLastDeclaration(CompoundStatement cStmt, Statement stmt)
  {
    FlatIterator iter = new FlatIterator(cStmt);
    Statement beforeStmt = null;
    while (iter.hasNext()) {
      try {
        beforeStmt = (Statement)iter.next(Statement.class);
      }
      catch(NoSuchElementException e) {
        break;
      }

      if(!(beforeStmt instanceof DeclarationStatement))
        break;
    }
    if(beforeStmt == null)
      cStmt.addStatement(stmt);
    else
      cStmt.addStatementBefore(beforeStmt, stmt);
  }

  public static boolean isSyncThreadStatement(Statement currStmt)
  {
    if (!(currStmt instanceof ExpressionStatement))
      return false;
    Expression expr = ((ExpressionStatement)currStmt).getExpression();
    if (!(expr instanceof FunctionCall))
      return false;
    FunctionCall fCall = (FunctionCall)expr;
    if ((fCall.getProcedure().getSymbolName()).equals("__syncthreads"))
      return true;
    return false;
  }

  public static FunctionCall createWrapperTransferFunction(FunctionCall core, int numCores,
      Procedure currentProc,
      LinkedList<Declaration> newParameters,
      LinkedList<Expression> newArgs,
      LinkedList<IDExpression> newArgIDs,
      boolean createFunctionDefn)
  {
    String name = core.getName() + "_wrapper";
    int numArguments = core.getNumArguments();
    if (numArguments*numCores != newParameters.size()) {
      System.out.println("Mismatch between wrapper and transfer function args");
    }

    CompoundStatement procBody = new CompoundStatement();
    Iterator<IDExpression> iter = newArgIDs.iterator();
    LinkedList<Expression> singleArgSet = new LinkedList<Expression>();
    int count = 0;
    for (int i=0;i<numCores;++i) {
      count = 0;
      singleArgSet.clear();
      while(count < numArguments) {
        IDExpression currId = iter.next();
        singleArgSet.add((IDExpression)currId.clone());
        ++count;
      }

      FunctionCall singleCall = (FunctionCall)core.clone();
      singleCall.setArguments(singleArgSet);

      procBody.addStatement(new ExpressionStatement(singleCall));
    }

    HashSet<Integer> commonIndex = new HashSet<Integer>();
    commonIndex.addAll(FCUDAGlobalData.getCommonArgsIndex(core));

    LinkedList<Declaration> tmpParameters = new LinkedList<Declaration>();
    LinkedList<Expression> tmpArgs = new LinkedList<Expression>();
    tmpParameters.addAll(newParameters);
    tmpArgs.addAll(newArgs);

    newParameters.clear();
    newArgs.clear();

    count = 0;
    int index = 0;
    while(count < tmpParameters.size()) {
      index = count%numArguments;
      if ((!commonIndex.contains(index) || count < numArguments) &&
          !newArgs.contains(tmpArgs.get(count)))  // This handles arguments that have not been replicated, e.g. kernel arguments
      {
        newParameters.add(tmpParameters.get(count));
        newArgs.add(tmpArgs.get(count));
      }
      ++count;
    }

    if(createFunctionDefn) {
      ProcedureDeclarator procDecl = new ProcedureDeclarator(new NameID(name), newParameters);
      Procedure wrapperProc = new Procedure(Specifier.VOID, procDecl, procBody);

      TranslationUnit parentTU = (TranslationUnit)currentProc.getParent();
      parentTU.addDeclarationBefore(currentProc, wrapperProc);
    }
    FunctionCall wrapperCall = new FunctionCall(new NameID(name), newArgs);
    return wrapperCall;
  }

  public static Expression getBidxID(int blockId, int dimId)
  {
    Expression bidX = MCUDAUtils.Bidx.getId(dimId);
    NameID id = new NameID(MCUDAUtils.Bidx.getString()+DuplicateForFCUDA.getPostFix(blockId)); //*AP* Check NameID usage
    IRTools.replaceAll((Traversable)bidX, MCUDAUtils.Bidx.getId().get(0), id);
    return bidX;
  }

  public static VariableDeclaration duplicateDeclaration(IDExpression refId, IDExpression newId)
  {
    SymbolTable st = FCUDAutils.getSymbolTableUnchecked(refId);
    if (st == null) {
      System.out.println("Could not find symbol table for "+refId.toString());
      System.exit(0);
    }
    return duplicateDeclaration(refId, newId, st);
  }

  public static VariableDeclaration duplicateDeclaration(IDExpression refId, IDExpression newId, SymbolTable st)
  {
    if (st == null) {
      System.out.println("Null symbol table ");
      System.exit(0);
    }
    VariableDeclaration vDecl = FCUDAutils.getVariableDeclarationUnchecked(refId, st);
    if(vDecl == null) {
      System.out.println("Could not find declaration for "+refId.toString());
      System.exit(0);
    }

    VariableDeclarator vDec = FCUDAutils.getDeclaratorFromDeclaration(vDecl, refId);

    VariableDeclaration newVarDecl = new VariableDeclaration(vDecl.getSpecifiers());
    VariableDeclarator newV = new VariableDeclarator(vDec.getSpecifiers(),
        (IDExpression)newId.clone(),
        vDec.getTrailingSpecifiers());
    newVarDecl.addDeclarator(newV);
    return newVarDecl;
  }

  public static Specifier getDataType(VariableDeclaration varDecl)
  {
    List<Specifier> specList = varDecl.getSpecifiers();
    for (Specifier currSpec : specList) {
      if (currSpec.isCType())
        return currSpec;
    }
    for (Specifier currSpec : specList) {
      if(currSpec instanceof UserSpecifier)
        return currSpec;
    }
    System.out.println("Could not find data type of "+varDecl.toString());
    System.exit(0);
    return null;
  }


  public static Statement getNxtStmt(Statement stmt)
  {
    List<Traversable> chi = stmt.getParent().getChildren();
    int index = Tools.indexByReference(chi, stmt);
    if (chi.size() > index)
      return (Statement) chi.get(index+1);
    else
      return null;
  }

  public static void parseList(String ptrString, LinkedList<String> tmpList)
  {
    int i;
    if (ptrString.charAt(0) != '[' || ptrString.charAt(ptrString.length()-1) != ']')
      Tools.exit("Pointer list should be of form \"[a1|a2 .. |aN]\" "+ptrString);
    ptrString = ptrString.substring(1, ptrString.length()-1);
    String[] result	= ptrString.split("\\|");
    for (i=0;i<result.length;++i) {
      tmpList.add(result[i]);
    }
  }
} // FCUDAutils
