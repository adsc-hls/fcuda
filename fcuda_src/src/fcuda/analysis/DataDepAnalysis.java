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

package fcuda.analysis;

import java.util.*;

import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.*;

import fcuda.utils.*;
import fcuda.*;

public class DataDepAnalysis
{
  public static int THREADIDX_DEP = 0x1;
  public static int POINTER_DEP = 0x2;
  public static int BLOCKIDX_DEP = 0x4;

  protected HashMap<IDExpression, TreeSet<Expression>> thrDefIDs;    // IDExpressions that are directly defined as expresions of threadIdx
  protected HashMap<IDExpression, TreeSet<Expression>> thrDepIDs;    // IDExpressions that are indirectly dependent on threadIdx

  private Procedure mProcedure;
  private HashMap<IDExpression, HashSet<Expression>> mVar2DefExprs;

  //Dep set has IDExpression - might have scalars, dim3 structs, pointer/array names
  private HashMap<IDExpression, HashSet<Expression>> mVar2Var;

  //ptrName to coeffs for burst transfer
  //ptr[X+c1*....]
  private HashMap<String, Expression[]> mPtr2BaseTerms;


  //array access to coeffs for burst transfer
  private HashMap<ArrayAccess, Expression[]> mPtrAccess2Coeffs;


  public static int MAX_COEFF_NUM=6;
  public static int NUMBER_BASE_TERMS=2;
  public static int C2_COEFF_INDEX=2;
  public static int C4_COEFF_INDEX=4;

  public DataDepAnalysis(Procedure proc)
  {
    thrDefIDs = new HashMap<IDExpression, TreeSet<Expression>>();
    thrDepIDs = new HashMap<IDExpression, TreeSet<Expression>>();
    mVar2DefExprs = new HashMap<IDExpression, HashSet<Expression>>();
    mVar2Var = new HashMap<IDExpression, HashSet<Expression>>();
    mPtr2BaseTerms = new HashMap<String, Expression[]>();
    mPtrAccess2Coeffs = new HashMap<ArrayAccess, Expression[]>();
    clear();

    mProcedure = proc;

    // Find variables directly defined by threadIdx members
    findDim3DefinedVars("threadIdx");
    Tools.printlnStatus("ThreadIdx defined Vars: "+thrDefIDs.toString(), 1);


    // Find variables in-directly defined by threadIdx members
    findDim3DependentVars();
    Tools.printlnStatus("ThreadIdx dependent Vars: "+thrDepIDs.toString(), 1);

    //if there are statements such as tx=threadIdx.x remove them and use threadIdx.x everywhere
    // *AP* This is currently necessary for FCUDA but it should be removed shortly
    if(Driver.getOptionValue("Fcuda") != null)
      removeDim3Placeholders();

    prepareMaps((Traversable)mProcedure);

    //Add threadIdx, blockIdx, blockDim as arguments
    //addDim3Argument();

    //Check if enclosing control statements contain threadIdx dep ctrl flow
    //checkForThreadDepCtrlFlow();

    //??
    //transformControlFlow();

  }

  public void clear()
  {
    mVar2DefExprs.clear();
    mVar2Var.clear();
    mPtr2BaseTerms.clear();
    mPtrAccess2Coeffs.clear();
  }

  public void addThrDefID(IDExpression id, Expression thrDef)
  {
    if(thrDefIDs.containsKey(id))
      (thrDefIDs.get(id)).add(thrDef);
    else {
      TreeSet<Expression> thrDefSet = new TreeSet<Expression>();
      thrDefSet.add(thrDef);
      thrDefIDs.put(id, thrDefSet);
    }
  }


  /*
     public Expression getThrDefExp(IDExpression id)
     {
     TreeSet<Expression> thrDefSet = null;
     Expression thrDefExp = null;

     if(thrDefIDs.containsKey(id)) {
     thrDefSet = thrDefIDs.get(id);
     if(thrDefSet.size() > 1) {
     for (Expression exp : thrDefSet)
     Tools.printlnStatus(" Def: "+exp.toString(), 0);
     Tools.exit(id.toString()+" has more than one ThreadIdx definitions: NOT HANDLED YET! ");
     }
     thrDefExp = (Expression) thrDefSet.first();
     }

     return thrDefExp;
     }
     */
  public boolean isThrDefExp(IDExpression id)
  {
    return thrDefIDs.containsKey(id);
  }

  public void addThrDepID(IDExpression id, Expression thrDep)
  {
    if(thrDepIDs.containsKey(id))
      (thrDepIDs.get(id)).add(thrDep);
    else {
      TreeSet<Expression> thrDepSet = new TreeSet<Expression>();
      thrDepSet.add(thrDep);
      thrDepIDs.put(id, thrDepSet);
    }
  }

  /*
     public Expression getThrDepExp(IDExpression id)
     {
     TreeSet<Expression> thrDepSet = null;
     Expression thrDepExp = null;

     if(thrDepIDs.containsKey(id)) {
     thrDepSet = thrDepIDs.get(id);
     if(thrDepSet.size() > 1) {
     for (Expression exp : thrDepSet)
     Tools.printlnStatus(" Dep: "+exp.toString(), 0);
     Tools.exit(id.toString()+" has more than one ThreadIdx dependent definitions: NOT HANDLED YET! ");
     }
     thrDepExp = (Expression) thrDepSet.first();
     }

     return thrDepExp;
     }
     */
  public boolean isThrDepExp(IDExpression id)
  {
    return thrDepIDs.containsKey(id);
  }

  public void printVar2Var()
  {
    System.out.println("mVar2Var:\n"+mVar2Var.toString());
  }

  public IDExpression getDefExprId(Expression defExpr)
  {
    //What can be defined in CUDA kernel?
    //Array or scalar - no dim3 variables
    //FIXME: Currently always map dependences for the top enclosing container (array/structure)
    // **AP** Should we also set dependences for lower-level containers???

    Expression defExpr2 = defExpr;
    Expression defExprId = null;
    while(true) {

      if (defExpr2 instanceof UnaryExpression) {
        defExprId = ((UnaryExpression)defExpr2).getExpression();
      }

      if(defExpr2 instanceof ArrayAccess)
        defExprId = ((ArrayAccess)defExpr2).getArrayName();

      if(defExpr2 instanceof AccessExpression)
        defExprId = ((AccessExpression)defExpr2).getLHS();

      if(defExpr2 instanceof IDExpression)
        defExprId = defExpr2;

      if(defExprId == null)
        Tools.exit("Could not determine IDExpression for "+defExpr.toString());

      if(defExprId instanceof IDExpression)
        break;

      defExpr2 = defExprId;
      defExprId = null;
    }

    return (IDExpression)defExprId;

  }


  // If tx=threadIdx.x
  // Remove this from program use threadIdx.x everywhere
  private void removeDim3Placeholders()
  {
    DepthFirstIterator iter = new DepthFirstIterator(mProcedure);
    iter.pruneOn(Expression.class);

    while(iter.hasNext())
    {
      AssignmentExpression aExpr = null;
      try
      {
        //*cetus-1.1*  aExpr = iter.next(AssignmentExpression.class);
        aExpr = (AssignmentExpression)iter.next(AssignmentExpression.class);
      }
      catch (NoSuchElementException e)
      {
        break;
      }

      if(aExpr.getOperator() == AssignmentOperator.NORMAL)
      {
        Expression tmpExpr = aExpr.getRHS();
        if(tmpExpr instanceof AccessExpression && aExpr.getLHS() instanceof IDExpression)
        {
          Expression placeholder = aExpr.getLHS();
          AccessExpression accExpr = (AccessExpression)tmpExpr;
          if(accExpr.getLHS() instanceof IDExpression)
          {
            IDExpression structId = (IDExpression)accExpr.getLHS();
            if(FCUDAutils.isCudaDim3Struct(structId))
            {
              /* *TAN* we need to thorough check here
               *  what if a var is assigned to a CUDADim3
               *  and then later assigned to another value?
               *  Therefore, we only remove vars that are
               *  CUDADim3 throughout the entire proc
               */
              Statement declStmt = FCUDAutils.getClosestParentStmt(FCUDAutils.
                  getVariableDeclarator((IDExpression)placeholder));
              Statement assignStmt = FCUDAutils.getClosestParentStmt(aExpr);
              List<Expression> listPlaceHolder = IRTools.findExpressions((Traversable)mProcedure, placeholder);
              boolean isConstantCUDADim3 = true;
              for (Expression ph : listPlaceHolder)
              {
                Traversable t = ph.getParent();
                if (!(t instanceof AssignmentExpression))
                  continue;
                AssignmentExpression phExpr = (AssignmentExpression)t;
                if (!(phExpr.getLHS().toString().equals(placeholder.toString())))
                  continue;
                ExpressionStatement assignStmt1 = new ExpressionStatement(phExpr.clone());
                if (!assignStmt1.toString().equals(assignStmt.toString()))
                  isConstantCUDADim3 = false;
              }
              if (isConstantCUDADim3)
              {
                declStmt.detach();
                assignStmt.detach();
                Tools.replaceAll((Traversable)mProcedure, placeholder, tmpExpr);
              }
            }
          }
        }
      }
    }
  }


  // Find variables immediately defined with dim3 variables
  // Hint: Use empty string for all implicit CUDA dim3 variables
  private void findDim3DefinedVars(String dim3Var)
  {
    DepthFirstIterator iter = new DepthFirstIterator(mProcedure);
    iter.pruneOn(Expression.class);

    while(iter.hasNext()) {
      boolean dim3RHS = false;
      AssignmentExpression aExpr = null;
      try {
        //*cetsu-1.1*  aExpr = iter.next(AssignmentExpression.class);
        aExpr = (AssignmentExpression)iter.next(AssignmentExpression.class);
      }
      catch (NoSuchElementException e) {
        break;
      }

      //*cetus-1.1*  Set<Expression> defSet = DataFlow.mayDefine(aExpr);
      Set<Expression> defSet = DataFlowTools.getDefSet(aExpr);
      //*cetus-1.1*  Set<Expression> useSet = DataFlow.getUseSet(aExpr.getRHS());
      Set<Expression> useSet = DataFlowTools.getUseSet(aExpr.getRHS());

      for(Expression expr : useSet) {
        if(FCUDAutils.isCudaDim3Struct(expr, dim3Var)) {
          dim3RHS = true;
          break;
        }
      }

      if(dim3RHS) {
        // Can not handle Arrays or structures or other weird things
        // FIXME ???
        /*
           if(defSet.size() > 1) {
           Tools.printlnStatus("WARNING: Can not handle non-scalar threadIdx-based definitions (1):", 0);
           Tools.printlnStatus(defSet.toString(), 0);
           continue;
           }
           */

        for(Expression def : defSet) {
          IDExpression defid = getDefExprId(def);
          if(defid != null)
            addThrDefID(defid, aExpr.getRHS());
          else {
            Tools.printlnStatus("WARNING: Could not find ID of threadIdx-based definition:", 0);
            Tools.printlnStatus(def.toString(), 0);
            continue;
          }
        }
      }
    }
  }

  //Check if var is dependant on threadIdx.
  public boolean isThrDepVar(IDExpression var)
  {
    return (thrDefIDs.containsKey(var) || thrDepIDs.containsKey(var));
  }

  // Find variables indirectly dependent from dim3 variables
  private void findDim3DependentVars()
  {
    DepthFirstIterator iter = new DepthFirstIterator(mProcedure);
    iter.pruneOn(Expression.class);

    while(iter.hasNext()) {
      boolean depRHS = false;
      AssignmentExpression aExpr = null;
      try {
        //*cetus-1.1*   aExpr = iter.next(AssignmentExpression.class);
        aExpr = (AssignmentExpression)iter.next(AssignmentExpression.class);
      }
      catch (NoSuchElementException e) {
        break;
      }

      //*cetus-1.1*   Set<Expression> defSet = DataFlow.mayDefine(aExpr);
      Set<Expression> defSet = DataFlowTools.getDefSet(aExpr);
      //*cetus-1.1*   Set<Expression> useSet = DataFlow.getUseSet(aExpr.getRHS());
      Set<Expression> useSet = DataFlowTools.getUseSet(aExpr.getRHS());

      for(Expression expr : useSet) {
        if(!(expr instanceof IDExpression))
          continue;
        if(isThrDefExp((IDExpression)expr) || isThrDepExp((IDExpression)expr)) {
          depRHS = true;
          break;
        }
      }

      if(depRHS) {
        // Can not handle Arrays or structures or other weird things
        // FIXME ???
        /*
           if(defSet.size() > 1) {
           Tools.printlnStatus("WARNING: Can not handle non-scalar threadIdx-dependent definitions (1):", 0);
           Tools.printlnStatus(defSet.toString(), 0);
           continue;
           }
           */

        for(Expression defid : defSet)
          addThrDepID(getDefExprId(defid), aExpr.getRHS());
      }
    }
  }



  // Get Definition expressions for varId
  HashSet<Expression>  getDefExpr(IDExpression varId) {

    HashSet<Expression> defExprs = null;
    if(mVar2DefExprs.containsKey(varId) )
      defExprs = mVar2DefExprs.get(varId);

    return defExprs;
  }


  public void prepareMaps(Traversable rootNode)
  {
    //DepthFirstIterator iter = new DepthFirstIterator(mProcedure);
    DepthFirstIterator iter = new DepthFirstIterator(rootNode);
    iter.pruneOn(Expression.class);

    //boolean flag = true;
    while(iter.hasNext())
    {
      Expression expr = null;
      try
      {
        //*cetus-1.1*  expr = iter.next(Expression.class);
        expr = (Expression)iter.next(Expression.class);
      }
      catch (NoSuchElementException e)
      {
        break;
      }

      //*cetus-1.1*  TreeSet<Expression> tmpSet = DataFlow.mayDefine((Traversable)expr);
      TreeSet<Expression> tmpSet = new TreeSet<Expression>();
      //*cetus-1.1*  tmpSet.addAll(DataFlow.defines((Traversable)expr));
      tmpSet.addAll(DataFlowTools.getDefSet((Traversable)expr));

      int priority = 4;
      if(tmpSet.size() > 0)
        priority = 1;

      Tools.printlnStatus("Expression "+expr.toString() + " defines:\n" + tmpSet.toString(), priority);

      for(Expression defExpr : tmpSet)
      {
        IDExpression defExprId = getDefExprId(defExpr);

        HashSet<Expression> set1;
        if(!mVar2DefExprs.containsKey((IDExpression)defExprId))
          set1 = new HashSet<Expression>();
        else
          set1 = mVar2DefExprs.get((IDExpression)defExprId);
        set1.add(expr);
        mVar2DefExprs.put((IDExpression)defExprId, set1);
      }
    }

    HashSet<IDExpression> scalarSet = new HashSet<IDExpression>();
    scalarSet.clear();

    DepthFirstIterator iter2 = new DepthFirstIterator(rootNode);
    iter2.pruneOn(IDExpression.class);
    while(iter2.hasNext())
    {
      IDExpression tmpIdExpr = null;
      try
      {
        //*cetus-1.1*    tmpIdExpr = iter2.next(IDExpression.class);
        tmpIdExpr = (IDExpression)iter2.next(IDExpression.class);
      }
      catch (NoSuchElementException e)
      {
        break;
      }
      VariableDeclarator tmp4 = FCUDAutils.getVariableDeclaratorUnchecked(tmpIdExpr);
      //if(tmp4 != null && FCUDAutils.isScalar(tmp4))
      if(tmp4 != null)
        scalarSet.add(tmpIdExpr);
    }

    HashSet<Expression> examinedSet = new HashSet<Expression>();
    HashSet<Expression> tmpSet;
    HashSet<Expression> toAnalyzeSet = new HashSet<Expression>();

    for(IDExpression currId : scalarSet)
    {
      examinedSet.clear();
      toAnalyzeSet.clear();
      toAnalyzeSet.add(currId);
      buildUseSet(examinedSet, toAnalyzeSet);

      tmpSet = new HashSet<Expression>();
      tmpSet.addAll(examinedSet);
      mVar2Var.put(currId, tmpSet);

      //System.out.println("For var "+currId.toString()+" input variables: "+examinedSet.toString());
    }

    examinedSet.clear();
    toAnalyzeSet.clear();
    scalarSet.clear();
  }

  private void buildUseSet(Set<Expression> examinedSet, Set<Expression> toAnalyzeSet)
  {
    boolean first = true;
    while(true)
    {
      if(toAnalyzeSet.size() > 0)
      {
        Iterator<Expression> setIter = toAnalyzeSet.iterator();
        // **AP** Basically each time only take the first element in toAnalyzeSet
        IDExpression currIdExpr = (IDExpression)setIter.next();
        toAnalyzeSet.remove(currIdExpr);
        if(!first)
          examinedSet.add(currIdExpr);
        buildUseSet(currIdExpr, examinedSet, toAnalyzeSet);
      }
      else
        break;
      first = false;
    }
  }

  private void buildUseSet(IDExpression currIdExpr, Set<Expression> examinedSet, Set<Expression> toAnalyzeSet)
  {
    Set<Expression> defSet = mVar2DefExprs.get(currIdExpr);
    if(defSet == null)
      return;
    for(Expression defExpr : defSet)
    {
      //*cetus-1.1*   Set<Expression> useSet = DataFlow.getUseSet(defExpr);
      Set<Expression> useSet = DataFlowTools.getUseSet(defExpr);

      //if(FCUDAutils.isArrayOrPointer(currIdExpr))
      //System.out.println("For var : "+currIdExpr+" def expr "+defExpr.toString()+ " Use set "+useSet.toString());

      //Iterator useIter = useSet.iterator();
      for(Expression useExpr : useSet)
      {
        //Expression useExpr = (Expression)useIter.next();

        //Don't analyze any CUDA dim3 structures
        if(useExpr instanceof AccessExpression && FCUDAutils.isCudaDim3Struct(useExpr))
        {
          examinedSet.add(((AccessExpression)useExpr).getLHS());
          continue;
        }
        if(useExpr instanceof IDExpression)
        {
          //Analyze array deps also
          //if(FCUDAutils.isArrayOrPointer((IDExpression)useExpr))
          //{
          //examinedSet.add(useExpr);
          //continue;
          //}

          //Analysis already done
          if(mVar2Var.containsKey(useExpr))
          {
            examinedSet.addAll(mVar2Var.get((IDExpression)useExpr));
            examinedSet.add(useExpr);
            continue;
          }
          if(examinedSet.contains(useExpr) || toAnalyzeSet.contains(useExpr))
            continue;
          toAnalyzeSet.add(useExpr);
        }
      }
    }
  }

  // Checks if split possible
  // If any of the conditions are satisfied, then cannot split currently :(
  // Conditions checked:
  // 1. Check if any for loop has threadIdx dependent condition, step, init etc.
  // 2. Check if any for loop has variables dependent on any array/pointer access - FIXME
  // 3. Check if statement is within  switch or doWhile or while - feeling lazy to code for each case - FIXME
  public boolean checkIfSplitPossible(Statement stmt)
  {
    Traversable t = (Traversable) stmt;
    while(!(t instanceof Procedure))
    {
      //FIXME - simply more coding needed to handle these types of ctrl-flow statements
      if(t instanceof DoLoop || t instanceof SwitchStatement || t instanceof WhileLoop)
      {
        System.out.println("##### Statement : "+stmt.toString()+" enclosed within "+t.toString());
        return false;
      }

      if(t instanceof ForLoop)
      {
        ForLoop forLoopStmt = (ForLoop)t;
        boolean flag = false;

        flag |= checkDep((Traversable)forLoopStmt.getInitialStatement(), THREADIDX_DEP|POINTER_DEP);
        flag |= checkDep((Traversable)forLoopStmt.getCondition(), THREADIDX_DEP|POINTER_DEP);
        flag |= checkDep((Traversable)forLoopStmt.getStep(), THREADIDX_DEP|POINTER_DEP);

        if(flag)
        {
          System.out.println("\n##### Statement : "+stmt.toString()+" enclosed within "+forLoopStmt.toString()
              +"\nwhich has tid/pointer deps #####\n");
          return false;
        }
      }

      if(t instanceof IfStatement)
      {
        IfStatement ifStmt = (IfStatement)t;
        boolean flag = false;

        flag |= checkDep((Traversable)ifStmt.getControlExpression(), THREADIDX_DEP|POINTER_DEP);

        if(flag)
        {
          System.out.println("##### Statement : "+stmt.toString()+" enclosed within "+ifStmt.toString()
              +" which has tid/pointer deps");
          return false;
        }
      }

      t= t.getParent();
    }
    return true;
  }

  public boolean checkDep(Traversable t, int depEnum)
  {
    boolean tidDep = false;
    boolean ptrDep = false;

    if((depEnum & THREADIDX_DEP) > 0)
      tidDep = true;
    if((depEnum & POINTER_DEP) > 0)
      ptrDep = true;

    DepthFirstIterator iter = new DepthFirstIterator(t);
    iter.pruneOn(IDExpression.class);
    boolean depFlag = false;
    while(iter.hasNext())
    {
      IDExpression currIdExpr = null;
      try
      {
        //*cetus-1.1*  currIdExpr = iter.next(IDExpression.class);
        currIdExpr = (IDExpression)iter.next(IDExpression.class);
      }
      catch (NoSuchElementException e)
      {
        break;
      }

      depFlag |= (ptrDep && FCUDAutils.isArrayOrPointer(currIdExpr));
      depFlag |= (tidDep && FCUDAutils.isThreadIdx(currIdExpr));

      if(depFlag)
        return true;

      if(mVar2Var.containsKey(currIdExpr))
      {
        Set<Expression> useSet = mVar2Var.get(currIdExpr);
        //Iterator useIter = useSet.iterator();
        for(Expression currUseExpr : useSet)
        {
          //Expression currUseExpr = (Expression)useIter.next();
          if(currUseExpr instanceof IDExpression)
          {
            IDExpression toCheckExpr = (IDExpression)currUseExpr;
            depFlag |= (ptrDep && FCUDAutils.isArrayOrPointer(toCheckExpr));
            depFlag |= (tidDep && FCUDAutils.isThreadIdx(toCheckExpr));
            if(depFlag)
              return true;
          }
          else
            Tools.exit("Check dep - what kind of expression is this ?"+currUseExpr.toString());
        }
      }
    }
    return depFlag;
  }

  // For all ptrs marked for burst transfer - check the address form
  // Form  : ptr[X + c1*(threadIdx.y + c2) + c3*(threadIdx.x + c4) + c5]
  // X,c1,c2,c3,c4,c5 are expressions independent of threadIdx
  //
  // burst is assumed to be from X+c1*threadIdx.y to X+c1*threadIdx.y+<size>  (specified in pragma)
  //
  // for(threadIdx.y=0;threadIdx.y < blockDim.y;threadIdx.y++)
  // 	memcpy(BRAM[threadIdx.y], ptr+X+c1*threadIdx.y, size);
  //
  // Any array access in kernel is replaced by: BRAM[threadIdx.y+c2][threadIdx.x+c4+c5]
  //
  // Returns - Expression representing X - if legal else null
  // -------------------------------------------------------------
  public Expression analyzeAccess(Statement transferStmt, IDExpression ptrId)
  {
    // *AP*		DepthFirstIterator iter = new DepthFirstIterator((Traversable)mProcedure);
    DepthFirstIterator iter = new DepthFirstIterator((Traversable)transferStmt);
    iter.pruneOn(ArrayAccess.class);

    Expression returnBaseExpr = null;
    Expression currentC1Expr = null;
    Expression baseExpr = null;
    Expression c1 = null, c2 = null, c3 = null;

    while(iter.hasNext())
    {
      ArrayAccess currAccess = null;
      try
      {
        //*cetus-1.1*  currAccess = iter.next(ArrayAccess.class);
        currAccess = (ArrayAccess)iter.next(ArrayAccess.class);
      }
      catch (NoSuchElementException e)
      {
        break;
      }
      if(currAccess.getArrayName() instanceof IDExpression)
      {
        //System.out.println("W "+(
        //((IDExpression)currAccess.getArrayName()).getSymbol().getSymbolName())
        //+ " "+ptrId.getSymbol().toString()
        //);

        //*cetus-1.1*   if(ptrId.getSymbol().getSymbolName().equals(((IDExpression)currAccess.getArrayName()).getSymbol().getSymbolName()))
        if(ptrId.equals((IDExpression)currAccess.getArrayName()))
        {
          //FIXME: only 1D pointer supported
          /*if(currAccess.getNumIndices() != 1)
            {
            System.out.println("Array access, but dimension greater than 1 "+currAccess.toString());
            return null;
            }*/
          Expression indexExpr = currAccess.getIndex(currAccess.getNumIndices() - 1);
          Expression[] indexPartsList = new Expression[MAX_COEFF_NUM];
          for(int i=0;i<MAX_COEFF_NUM;++i)
            indexPartsList[i] = null;
          System.out.println("Parsing "+indexExpr.toString());
          if(parseIndexExpr((Expression)indexExpr.clone(), indexPartsList, true))
          {
            //baseExpr = indexPartsList.get(0);
            baseExpr = indexPartsList[0];
            System.out.println("Base expr " + ((baseExpr == null) ? "null" : baseExpr.toString()));

            for(int i=1;i<MAX_COEFF_NUM;++i)
              System.out.println("c"+i+" = "+((indexPartsList[i] == null) ? "null" :
                    indexPartsList[i].toString()));

            if(returnBaseExpr != null && !baseExpr.equals(returnBaseExpr))
            {
              System.out.println("Array access, different base addr "+currAccess.toString());
              return null;
            }

            if(currentC1Expr != null && indexPartsList[1] != null
                && !indexPartsList[1].equals(currentC1Expr))
            {
              System.out.println("Array access, different C1 addr "+currAccess.toString());
              return null;
            }
            returnBaseExpr = baseExpr;
            currentC1Expr = indexPartsList[1];

            //Put X and c1 in the base expr for the burst

            // **AP** Removing the if condition to allow more transfers to/from the same pointer
            // **AP** Ideally if we want to keep older parameters we need to use BRAM+taskName
            // **AP** to differentiate entries in mPtr2BaseTerms
            //if(!mPtr2BaseTerms.containsKey(ptrId.getSymbol().getSymbolName()))
            {
              Expression[] baseTerms = new Expression[DataDepAnalysis.NUMBER_BASE_TERMS];
              baseTerms[0] = indexPartsList[0];
              baseTerms[1] = indexPartsList[1];
              //*cetus-1.1*   mPtr2BaseTerms.put(ptrId.getSymbol().getSymbolName(), baseTerms);
              mPtr2BaseTerms.put(ptrId.toString(), baseTerms);
            }

            mPtrAccess2Coeffs.put(currAccess, indexPartsList);
          }
          else
          {
            System.out.println("Array access, index does not satisfy required form "+currAccess.toString());
            return null;
          }
        }
      }
      else
      {
        System.out.println("Array access, but completely unhandled currently "+currAccess.toString());
        return null;
      }
    }
    return returnBaseExpr;
  }

  //FIXME: Extremely limited in the way it handles stuff
  public boolean parseIndexExpr(Expression indexExpr, Expression[] indexPartsList, boolean provideChance)
  {
    Expression[] tmpArray = new Expression[MAX_COEFF_NUM];
    for(int i=0;i<MAX_COEFF_NUM;++i)
      tmpArray[i] = null;

    Expression baseExpr = null;
    boolean returnVal = true;
    int count = 0;

    List<Expression> terms = FCUDAutils.getTerms(indexExpr);
    //Iterator termIter = terms.iterator();
    System.out.println("Terms "+terms.toString());
    for(Expression currentTerm : terms)
    {
      //Expression currentTerm = (Expression)termIter.next();
      if(!checkDep(currentTerm, THREADIDX_DEP))
      {
        indexPartsList[0] = (indexPartsList[0] == null) ? currentTerm :
          Symbolic.add(indexPartsList[0], currentTerm);
        continue;
      }

      if(FCUDAutils.isCudaDim3Struct(currentTerm, "threadIdx"))
      {
        //threadIdx.x
        if(currentTerm.equals(MCUDAUtils.Tidx.getId(0)))
          indexPartsList[3] = new IntegerLiteral(1);

        //threadIdx.y
        if(currentTerm.equals(MCUDAUtils.Tidx.getId(1)))
          indexPartsList[1] = new IntegerLiteral(1);
        continue;
      }

      if(provideChance && currentTerm instanceof BinaryExpression)
      {
        BinaryExpression binExpr = (BinaryExpression)currentTerm;
        Expression examineNode, cMulNode;
        if(binExpr.getOperator() == BinaryOperator.MULTIPLY)
        {
          if(checkDep(binExpr.getLHS(), THREADIDX_DEP))
          {
            if(!checkDep(binExpr.getRHS(), THREADIDX_DEP))
            {
              examineNode = binExpr.getLHS();
              cMulNode = binExpr.getRHS();
            }
            else
            {
              System.out.println("Both factors of term "+currentTerm.toString()+" are threadIdx dependent");
              return false;
            }
          }
          else
          {
            examineNode = binExpr.getRHS();
            cMulNode = binExpr.getLHS();
          }

          if(parseIndexExpr(examineNode, tmpArray, false))
          {
            if(tmpArray[3] != null)
            {
              indexPartsList[3] = cMulNode;
              indexPartsList[4] = tmpArray[0];
            }
            if(tmpArray[1] != null)
            {
              indexPartsList[1] = cMulNode;
              indexPartsList[2] = tmpArray[0];
            }
            for(int i=0;i<MAX_COEFF_NUM;++i)
              tmpArray[i] = null;
            continue;
          }
          else
          {
            System.out.println("Could not parse the expression "+examineNode);
            return false;
          }
        }
        else
        {
          System.out.println("Term "+currentTerm.toString()+" is not a multiply expression");
          return false;
        }
      }


      System.out.println("Term "+currentTerm.toString()+" cannot be handled");
      return false;
    }

    return true;
  }


  public Expression[] getBaseTerms(String ptrName)
  {
    if(!mPtr2BaseTerms.containsKey(ptrName))
      Tools.exit("No entry in ptr 2 coeff map for "+ptrName);
    return mPtr2BaseTerms.get(ptrName);
  }

  public Expression[] getAccessCoeffs(ArrayAccess ptr)
  {
    if(!mPtrAccess2Coeffs.containsKey(ptr))
      Tools.exit("No entry in ptr 2 coeff map for "+ptr);
    return mPtrAccess2Coeffs.get(ptr);
  }

  public Expression getCoeffC2(ArrayAccess ptr)
  {
    if(!mPtrAccess2Coeffs.containsKey(ptr))
      Tools.exit("No entry in ptr 2 coeff map for "+ptr);
    Expression[] arrayCoeffs = mPtrAccess2Coeffs.get(ptr);
    return arrayCoeffs[C2_COEFF_INDEX];
  }

  public Expression getCoeffC4(ArrayAccess ptr)
  {
    if(!mPtrAccess2Coeffs.containsKey(ptr))
      Tools.exit("No entry in ptr 2 coeff map for "+ptr);
    Expression[] arrayCoeffs = mPtrAccess2Coeffs.get(ptr);
    return arrayCoeffs[C4_COEFF_INDEX];
  }

  public Set<Expression> getVarDeps(IDExpression id)
  {
    HashSet<Expression> deps = null;
    if(mVar2Var.containsKey(id))
      deps = mVar2Var.get(id);

    return deps;
  }
}
