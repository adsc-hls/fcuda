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
import fcuda.ir.*;
import fcuda.utils.*;

import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.*;

/*
 * Array Partitioning
 *
 */
public class PartitionArrays extends KernelTransformPass 
{
  private class UnrollIDinfo
  {
    public int old_step;
    public int new_step;
    boolean isCtrlFlow;
    public Statement defStmt;
    public HashSet<Statement> mayDefStmt;
    public HashSet<Statement> useStmt;

    UnrollIDinfo(int step, int nstep, Statement dStmt) 
    {
      old_step = step;
      new_step = nstep;
      defStmt = dStmt;
      isCtrlFlow = false;
      mayDefStmt = new HashSet();
      useStmt = new HashSet();
    }
  }

  public static int KERN_PROC = 0x1;
  public static int COMP_PROC = 0x2;
  public static int TRANS_FETCH_PROC = 0x4;
  public static int TRANS_WRITE_PROC = 0x8;

  private Procedure mProcedure;

  private int numDims = 3;

  private boolean bconsUnroll;

  private int procEnum;

  private int mempartFactor;

  private List<String> shapeList;

  private int sizeMempart;

  private int maxStep;

  private int partDims;

  private ThreadLoop tloop;

  private boolean isUpdatedTLoop;

  private Expression loopIndex;

  private Expression loopBound;

  private List<String> mPartitionArrays;

  private String Array_Suffix = "_part";

  private String idxName = "uIdx";

  private HashMap<IDExpression, ArraySpecifier>mID2ArrSpec;

  private HashSet<Statement> mCoreStmts;

  private HashMap<IDExpression, UnrollIDinfo> mUnrollIDinfo;

  private HashMap<String, VariableDeclaration> mId2PartArrSym;

  public enum StatementNames 
  {
    DefaultStatement,
    ExpressionStatement,
    WhileLoop, DeclarationStatement,
    CompoundStatement,
    AnnotationStatement,
    ForLoop,
    ThreadLoopStmt,
    IfStatement
  }

  private StatementNames getStmtTypeId(Statement stmt) 
  {
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
      return StatementNames.ThreadLoopStmt;
    if (stmt instanceof IfStatement)
      return StatementNames.IfStatement;
    return StatementNames.DefaultStatement;
  }

  public String getPassName() 
  {
    return new String("[PartitionArrays-MCUDA]");
  }

  public PartitionArrays(Program program) 
  {
    super(program);
    mID2ArrSpec = new HashMap<IDExpression, ArraySpecifier>();
    mID2ArrSpec.clear();
    mCoreStmts = new HashSet<Statement>();
    mCoreStmts.clear();
    mUnrollIDinfo = new HashMap<IDExpression, UnrollIDinfo>();
    mId2PartArrSym = new HashMap<String, VariableDeclaration>();
    mId2PartArrSym.clear();
  }

  public boolean isStepFormat(Expression expr) 
  {
    String str = expr.toString();
    if (str.indexOf('+') != -1)
      return true;
    return false;
  }

  public boolean inArrayPartList(String name) 
  {
    Iterator iter = mPartitionArrays.iterator();
    while (iter.hasNext()) {
      String str = (String) iter.next();
      if (str.equals(name))
        return true;
    }
    return false;
  }

  public Expression createIndexPlusExpr(Expression index, Expression expr) 
  {
    Expression index0 = (Expression) index.clone();
    Expression new_expr = new BinaryExpression(index0, BinaryOperator.ADD, expr);
    return new_expr;
  }

  public Expression createSuffixExpr(Expression index, int factor) 
  {
    IDExpression ide = (IDExpression) index;
    String name = ide.toString() + Array_Suffix + factor;
    NameID id = new NameID(name);
    return id;
  }

  private Expression updateLoopIndexByMempart() 
  {
    Expression mul = new IntegerLiteral(mempartFactor);
    Expression exp = new BinaryExpression((Expression)loopIndex.clone(),
        BinaryOperator.MULTIPLY,
        mul);
    return exp;
  }

  public Expression createIndexPlus(Expression index, int factor) 
  {
    Expression index0 = (Expression) index.clone();
    Expression add = new IntegerLiteral(factor);
    Expression expr = new BinaryExpression(index0, BinaryOperator.ADD, add);
    return expr;
  }

  /*
   * 1. Only Arrays which are accessed in the pattern of A[exp + c] can be
   * partitioned, where exp is an expression with loop index and c is step
   * value. c is range of [0, maxStep].
   *
   * We need to find this maxStep first and we assume array accesses A[exp +
   * 0], ... A [exp + maxStep].
   */
  public void findMaxStep(Procedure proc) {
    maxStep = 0;

    DepthFirstIterator iter = new DepthFirstIterator(proc);
    while (iter.hasNext()) {
      Object child = iter.next();
      /* all the affected variables by loopIndex */

      if (child instanceof ArrayAccess) {
        ArrayAccess aa = (ArrayAccess) child;

        if (inArrayPartList(aa.getArrayName().toString())) {
          System.out.println("[ArrayAccess] " + aa.toString());

          List<Expression> list = aa.getIndices();
          Iterator it = list.iterator();
          while (it.hasNext()) {
            Expression exp = (Expression) it.next();

            if (isStepFormat(exp)) {
              System.out.println("[expr]" + exp.toString());
              FlatIterator fit = new FlatIterator(exp);
              while (fit.hasNext()) {
                Object obj = fit.next();
                if (obj instanceof IntegerLiteral) {
                  System.out.println(
                      "[int]" + obj.toString());
                  int val = (int) ((IntegerLiteral) obj).getValue();
                  if (val > maxStep)
                    maxStep = val;
                }
              }
            }
          }
        }
      }
    }

    sizeMempart = (maxStep + 1) / mempartFactor;
    System.out.println("[Max Step] ------ " + maxStep);
    System.out.println("[sizeMem ] ------ " + sizeMempart);

  }

  private void mempartDefaultStatement_consecutive(Statement stmt)
  {
    System.out.println("[DefaultStatement] " + stmt.toString());

    int step = 0;

    DepthFirstIterator iter = new DepthFirstIterator(stmt);
    while (iter.hasNext()) {
      Object child = iter.next();

      if (child instanceof ArrayAccess) {
        ArrayAccess aa = (ArrayAccess) child;
        /*
         * Tools.printlnStatus("[array access]" + aa.toString(), 0);
         * Tools.printlnStatus("[array name]" +
         * aa.getArrayName().toString(), 0);
         * Tools.printlnStatus("[array indice]" +
         * aa.getIndices().toString(), 0);
         */

        if (inArrayPartList(aa.getArrayName().toString())) {
          Expression arrayname = aa.getArrayName();
          List<Expression> list = aa.getIndices();
          Iterator it = list.iterator();
          while (it.hasNext()) {
            Expression exp = (Expression) it.next();
            if (isStepFormat(exp)) {
              // Tools.printlnStatus("[expr]" + exp.toString(),
              // 0);
              /*
                 FlatIterator fit = new FlatIterator(exp);
                 while (fit.hasNext()) {
                 Object obj = fit.next();
                 if (obj instanceof IntegerLiteral) {
              // Tools.printlnStatus("[int]" +
              // obj.toString(), 0);
              step = (int) ((IntegerLiteral) obj)
              .getValue();

              IntegerLiteral newint = new IntegerLiteral(
              step % sizeMempart);

              ((IntegerLiteral) obj).swapWith(newint);
              break;
                 }
                 }*/

              Expression rhs  = ((BinaryExpression) exp).getRHS();
              if (rhs instanceof IntegerLiteral) {
                //format of threadIdx.x + i --> threadIdx.x + i % sizeMempart.
                System.out.println("[threadIdx + i]");
                step = (int) ((IntegerLiteral) rhs).getValue();
                int new_step = step % sizeMempart;
                Expression new_expr = new IntegerLiteral(new_step);
                rhs.swapWith(new_expr);
              } else {
                //format of threadIdx.x + i + offset --> threadIdx.x + i % sizeMempart + offset/mempaFactor.
                System.out.println("[threadIdx + i + offset]" + rhs.toString());
                Expression new_rhs = new BinaryExpression((Expression) rhs.clone(),
                    BinaryOperator.DIVIDE, new IntegerLiteral(mempartFactor));
                rhs.swapWith(new_rhs);
                Expression lhs  = ((BinaryExpression) exp).getLHS();
                Expression rhs0 = ((BinaryExpression) lhs).getRHS();
                step = (int) ((IntegerLiteral) rhs0).getValue();
                int new_step = step % sizeMempart;
                rhs0.swapWith(new IntegerLiteral(new_step));

                //Tools.replaceAll((Traversable)exp, rhs0, new IntegerLiteral(new_step));
                //Tools.replaceAll((Traversable)exp, rhs,  new_rhs);
              }
              break;
            }
          }

          ArrayAccess newaa = new ArrayAccess(createSuffixExpr(
                arrayname, step / sizeMempart), list);
          aa.swapWith(newaa);
        }
      } else if (child.toString().equals(loopIndex.toString())) {
        Boolean arrayChild = false;
        Expression exp = (Expression) child;
        Traversable t = exp.getParent();
        while (t != stmt) {
          System.out.println("[parent is]" + t.toString());
          if (t instanceof ArrayAccess) {
            arrayChild = true;
            break;
          }
          t = t.getParent();

        }
        // update only if it is not an array indices.
        if (!arrayChild)
          exp.swapWith(updateLoopIndexByMempart());
      }
    }
  }

  private void mempartDefaultStatement_non_consecutive(Traversable stmt) 
  {
    System.out.println("[DefaultStatement] " + stmt.toString());
    int stepTot = -1;
    Procedure proc = (Procedure) Tools.getAncestorOfType(stmt, Procedure.class);
    TreeMap<ArrayAccess, Integer> mAA2uIdx = new TreeMap<ArrayAccess, Integer>();
    mAA2uIdx.clear();
    TreeMap<Statement, Statement> mThrDepDefs = new TreeMap<Statement, Statement>();
    mThrDepDefs.clear();

    // First: Check if statement is enclosed in thread-controlled if statement
    IfStatement parentIfStmt = (IfStatement)Tools.getAncestorOfType(stmt, IfStatement.class);
    if (parentIfStmt != null) {
      Expression ifControl = parentIfStmt.getControlExpression();
      Set<Expression> ctrlVars = DataFlow.getUseSet(ifControl);
      for (Expression ctrlVar : ctrlVars) {
        if (ctrlVar instanceof IDExpression)
          stepTot = FCUDAGlobalData.isUnrolledID(proc.getName(), (IDExpression) ctrlVar);
        if (stepTot >= 0)
          break; // Stop on first unrolled variable - remaining should use same unroll index (probably?)
      }
    }


    // Second: Go through statements variables and check if they need update
    DepthFirstIterator iter = new DepthFirstIterator(stmt);
    while (iter.hasNext()) {
      Object useExp = iter.next();
      boolean isArrayPartition = false;
      // Handle IDExpression
      if (useExp instanceof IDExpression) {
        int step = FCUDAGlobalData.isUnrolledID(proc.getName(), (IDExpression) useExp);
        System.out.println("useExp: " + useExp.toString() + "  uIdx: " + step);

        if (step >= 0) {
          if (shapeList != null) {
            int shape = Integer.parseInt(shapeList.get(0));
            if (shape != 0 && loopBound != null)
              Tools.replaceAll(stmt, loopBound, new BinaryExpression(new IntegerLiteral(shape),
                    BinaryOperator.DIVIDE,
                    new IntegerLiteral(mempartFactor)));
          }

          // Check if in ArrayAccess index
          ArrayAccess aa = (ArrayAccess) Tools.getAncestorOfType((Traversable)useExp, ArrayAccess.class);
          // *TAN* get the farthest ancestor of aa
          //while ((ArrayAccess) IRTools.getAncestorOfType((Traversable)aa, ArrayAccess.class) != null) {
          //aa = (ArrayAccess) IRTools.getAncestorOfType((Traversable)aa, ArrayAccess.class);
          //}

          if (aa != null) {
            if (inArrayPartList(aa.getArrayName().toString())) {
              isArrayPartition = true;
              System.out.println("mAA2uIdx(1):\n >>> "+mAA2uIdx.toString());
              if (mAA2uIdx.containsKey(aa)) {
                if (mAA2uIdx.get(aa) != step)
                  Tools.exit("Conflict in memory partitioning decision for array "+
                      aa.getArrayName().toString());
              } else {
                System.out.println("mAA2uIdx(2):\n >>> "+mAA2uIdx.toString());
                mAA2uIdx.put(aa, step);
              }
            }
          }
          // *******************************
          // If variable is a uIdx# variable
          Statement thrDepDef = (Statement) Tools.getAncestorOfType((Traversable) useExp, Statement.class);
          NameID idxCoef = new NameID(idxName+step);
          if (isArrayPartition && ((Expression)idxCoef).equals(useExp)) {
            int new_step = step % sizeMempart;
            if (new_step != step) {
              // Check if statement defines unrolled variable
              Set <Expression> defIDs = DataFlow.defines(thrDepDef);
              IDExpression unrollID = null;
              if (aa == null) { // uIdx# variable should not be in array index
                for (Expression defID : defIDs) {
                  if(defID instanceof IDExpression) {
                    if(FCUDAGlobalData.isUnrolledID(proc.getName(), (IDExpression) defID) >= 0) {
                      unrollID = (IDExpression)defID;
                      break;
                    }
                  }
                }
              }

              if (unrollID != null) { // defines other unrolled variable
                UnrollIDinfo uIDinf = new UnrollIDinfo(step, new_step, thrDepDef);
                mUnrollIDinfo.put(unrollID, uIDinf);
              } else {               // does not define other unrolled variable
                NameID new_expr = new NameID(idxName+new_step);
                Tools.replaceAll((Expression)useExp, (Expression)useExp, new_expr);
              }
            }
          } else {   // non uIdx# unrolled variable
            if (DataFlow.mayDefine(thrDepDef).contains((Expression)useExp)) {
              if (mUnrollIDinfo.keySet().contains((Expression)useExp))
                mUnrollIDinfo.get((IDExpression)useExp).mayDefStmt.add(thrDepDef);
            } else if (DataFlow.getUseSet(thrDepDef).contains((Expression)useExp)) {
              if (mUnrollIDinfo.keySet().contains((Expression)useExp)) {
                mUnrollIDinfo.get((IDExpression)useExp).useStmt.add(thrDepDef);
              }
            }
          }
        }
      }


      // Handle ArraysAccess without unrolled variables in their indices
      if ((stepTot >= 0) && (useExp instanceof ArrayAccess)) {
        ArrayAccess aAcc = (ArrayAccess) useExp;
        if (inArrayPartList(aAcc.getArrayName().toString())) {
          if (mAA2uIdx.containsKey(aAcc)) {
            System.out.println("mAA2uIdx(3):\n >>> "+mAA2uIdx.toString());
            if (mAA2uIdx.get(aAcc) != stepTot) {
              System.out.println("Conflict in memory partitioning decision for array " +
                  aAcc.getArrayName().toString());
              System.exit(0);
            }
          } else {
            System.out.println("mAA2uIdx(4):\n >>> " + mAA2uIdx.toString());
            mAA2uIdx.put(aAcc, stepTot);
          }
        }
      }

    } // while (iter.hasNext())

    System.out.println("mAA2uIdx(final):\n >>> "+mAA2uIdx.toString());

    // Third: Replace ArrayAccess with partitioned one
    for (Map.Entry<ArrayAccess, Integer> e : mAA2uIdx.entrySet()) {
      ArrayAccess oldaa = e.getKey();
      int step = e.getValue().intValue();
      System.out.println("    handling array access: "+oldaa.toString());
      System.out.println("             step=" + step);
      List<Expression> list = new LinkedList<Expression>();
      for (Expression aaIdx : oldaa.getIndices())
        list.add(aaIdx.clone());
      ArrayAccess newaa = new ArrayAccess(createSuffixExpr(oldaa.getArrayName(),
            step / sizeMempart),
          list);
      Tools.replaceAll(stmt, oldaa, newaa);

      System.out.println("[DefaultStatement(2)] " + stmt.toString());
    }
  } // mempartDefaultStatement_non_consecutive()

  public void updateFetchStatement(Statement stmt, CompoundStatement newBody) 
  {
    for (int i = 0; i < mempartFactor; i++) {
      Statement newS = (Statement) stmt.clone();
      ExpressionStatement exps = (ExpressionStatement) newS;
      Expression exp = exps.getExpression();

      /*
       * must be memory function call.
       */
      if (exp instanceof FunctionCall) {
        FunctionCall fun = (FunctionCall) exp;

        // Handle BRAM array renaming
        Expression dest  = fun.getArgument(0);

        Set<Expression> destUses = DataFlow.getUseSet(dest);
        for (Expression dExp : destUses) {
          if(inArrayPartList(dExp.toString())) {
            Expression newArr = createSuffixExpr(dExp, i);
            Tools.replaceAll(dest, dExp, newArr);
            break;
          }
        }

        // Handle POINTER Indexing
        Expression src   = fun.getArgument(1);
        Expression new_expr = null;
        if (bconsUnroll) {
          if (loopBound != null) {
            Expression expr = new BinaryExpression((Expression)loopIndex.clone(),
                BinaryOperator.MULTIPLY,
                new IntegerLiteral(mempartFactor));
            new_expr = createIndexPlus(expr, i);
          } else {
            Expression mcpySize = fun.getArgument(2);
            Expression expr = new BinaryExpression((Expression)((BinaryExpression)mcpySize).getLHS().clone(),
                BinaryOperator.MULTIPLY,
                new IntegerLiteral(mempartFactor));
            new_expr = createIndexPlus(expr, i);
          }
        } else {
          if(loopBound != null) {
            Expression offset1 = new BinaryExpression((Expression)loopBound.clone(),
                BinaryOperator.DIVIDE,
                new IntegerLiteral(mempartFactor));
            Expression offset2 = new BinaryExpression(offset1,
                BinaryOperator.MULTIPLY,
                new IntegerLiteral(i));
            new_expr = createIndexPlusExpr(loopIndex, offset2);
          } else {
            Expression mcpySize = fun.getArgument(2);
            Expression offset2 = new BinaryExpression((Expression)((BinaryExpression)mcpySize).getLHS().clone(),
                BinaryOperator.MULTIPLY,
                new IntegerLiteral(i));
            new_expr = createIndexPlusExpr(src, offset2);
          }
        }
        if (loopBound != null)
          Tools.replaceAll((Traversable)src, loopIndex, new_expr);
        else
          Tools.replaceAll(fun, src, new_expr);
      }
      newBody.addStatement(newS);
    }
  }

  public void updateWriteStatement(Statement stmt, CompoundStatement newBody) 
  {
    for (int i = 0; i < mempartFactor; i++) {
      Statement newS = (Statement) stmt.clone();
      ExpressionStatement exps = (ExpressionStatement) newS;
      Expression exp = exps.getExpression();

      /*
       * must be memory function call.
       */
      if (exp instanceof FunctionCall) {
        FunctionCall fun = (FunctionCall) exp;

        // Handle BRAM array renaming
        Expression   src = fun.getArgument(1);

        Set<Expression> srcUses = DataFlow.getUseSet(src);
        for (Expression sExp : srcUses) {
          if (inArrayPartList(sExp.toString())) {
            Expression newArr = createSuffixExpr(sExp, i);
            Tools.replaceAll(src, sExp, newArr);
            break;
          }
        }
        // Handle POINTER Indexing
        Expression dest   = fun.getArgument(0);
        Expression new_expr = null;
        if (bconsUnroll) {
          if(loopBound != null) {
            Expression expr = new BinaryExpression((Expression)loopIndex.clone(),
                BinaryOperator.MULTIPLY,
                new IntegerLiteral(mempartFactor));
            new_expr = createIndexPlus(expr, i);
          } else {
            Expression mcpySize = fun.getArgument(2);
            Expression expr = new BinaryExpression((Expression)((BinaryExpression)mcpySize).getLHS().clone(),
                BinaryOperator.MULTIPLY,
                new IntegerLiteral(mempartFactor));
            new_expr = createIndexPlus(expr, i);
          }
        } else  {
          if (loopBound != null) {
            Expression offset1 = new BinaryExpression((Expression)loopBound.clone(),
                BinaryOperator.DIVIDE,
                new IntegerLiteral(mempartFactor));
            Expression offset2 = new BinaryExpression(offset1,
                BinaryOperator.MULTIPLY,
                new IntegerLiteral(i));
            new_expr = createIndexPlusExpr(loopIndex, offset2);
          } else { // No Threadloop exists
            Expression mcpySize = fun.getArgument(2);
            Expression offset2 = new BinaryExpression((Expression)((BinaryExpression)mcpySize).getLHS().clone(),
                BinaryOperator.MULTIPLY,
                new IntegerLiteral(i));
            new_expr = createIndexPlusExpr(dest, offset2);

          }

          if (loopBound != null)
            Tools.replaceAll((Traversable)dest, loopIndex, new_expr);
          else
            Tools.replaceAll(fun, dest, new_expr);
        }
      }
      newBody.addStatement(newS);
    }
  }

  private void mempartFetchWrite(Statement stmt) 
  {
    System.out.println("[FetchWrite] " + stmt.toString());
    if (!(stmt instanceof ExpressionStatement))
      return;
    ExpressionStatement exps = (ExpressionStatement) stmt;
    Expression exp = exps.getExpression();

    /*
     * must be memory function call.
     */
    if (exp instanceof FunctionCall) {
      System.out.println("[Funcall]" + exp.toString());
      FunctionCall fun = (FunctionCall) exp;
      Expression dest = fun.getArgument(0);
      boolean isFetch = false;
      Set<Expression> destUses = DataFlow.getUseSet(dest);
      for (Expression dExp : destUses) {
        if (inArrayPartList(dExp.toString())) {
          isFetch = true;
          break;
        }
      }

      Expression src = fun.getArgument(1);
      boolean isWrite = false;
      Set<Expression> srcUses = DataFlow.getUseSet(src);
      for (Expression sExp: srcUses) {
        if (inArrayPartList(sExp.toString())) {
          isWrite = true;
          break;
        }
      }

      // fetch function
      if((procEnum & TRANS_FETCH_PROC) > 0) {
        // destination contains partioned array
        if (isFetch) {
          System.out.println(" ... is fetch!");
          if (tloop != null) {
            //isUpdatedTLoop = true;
            if (!isUpdatedTLoop) {
              System.out.println("[update loop bound/index]");
              Expression expr = new BinaryExpression((Expression)tloop.getBound(partDims).clone(),
                  BinaryOperator.DIVIDE,
                  new IntegerLiteral(mempartFactor));
              tloop.setBound(partDims, expr);
              isUpdatedTLoop = true;
            }
          } else {
            //Get size of memcpy
            Expression mcpySize = fun.getArgument(2);
            System.out.println("3rd arg class: "+mcpySize.getClass().toString());
            if (!(mcpySize instanceof BinaryExpression))
              Tools.exit("Don't know how to handle this" + mcpySize.toString());
            Expression newSize = new BinaryExpression((Expression)((BinaryExpression)mcpySize).getLHS().clone(),
                BinaryOperator.DIVIDE,
                new IntegerLiteral(mempartFactor));

             Tools.replaceAll(((BinaryExpression)mcpySize).getLHS(),
                ((BinaryExpression)mcpySize).getLHS(),
                newSize);
          }
          CompoundStatement newbody = new CompoundStatement();
          updateFetchStatement(stmt, newbody);
          stmt.swapWith(newbody);
        } else {
          System.out.println("hello " + stmt);
        }
      }

      // write function
      if ((procEnum & TRANS_WRITE_PROC) > 0) {
        if (isWrite) {
          if (tloop != null) {
            if (!isUpdatedTLoop) {
              Expression expr = new BinaryExpression((Expression)tloop.getBound(partDims).clone(),
                  BinaryOperator.DIVIDE,
                  new IntegerLiteral(mempartFactor));
              tloop.setBound(partDims, expr);
              isUpdatedTLoop = true;
            }
          } else {
            //Get size of memcpy
            Expression mcpySize = fun.getArgument(2);
            System.out.println("3rd arg class: "+mcpySize.getClass().toString());
            if (!(mcpySize instanceof BinaryExpression))
              Tools.exit("Don't know how to handle this" + mcpySize.toString());
            Expression newSize = new BinaryExpression((Expression)((BinaryExpression)mcpySize).getLHS().clone(),
                BinaryOperator.DIVIDE,
                new IntegerLiteral(mempartFactor));

            ((BinaryExpression)mcpySize).getLHS().swapWith(newSize);
          }

          CompoundStatement newbody = new CompoundStatement();
          updateWriteStatement(stmt, newbody);
          stmt.swapWith(newbody);
        }
      }
    }
  }

  private void mempartDefaultStatement_wrap(Statement stmt) 
  {
    if ((procEnum & COMP_PROC) > 0) {
      if (bconsUnroll)
        mempartDefaultStatement_consecutive(stmt);
      else
        mempartDefaultStatement_non_consecutive(stmt);
    }
    if ( (procEnum & TRANS_FETCH_PROC) > 0 || (procEnum & TRANS_WRITE_PROC) > 0) {
      mempartFetchWrite(stmt);
    }
  }

  private ArraySpecifier partArrDims(Declarator declor) {
    ArraySpecifier aSpec = null;
    IDExpression idexp = declor.getID();

    if (mID2ArrSpec.containsKey(idexp))
      aSpec = mID2ArrSpec.get(idexp);
    else {
      List<Specifier> arrDimSpec = declor.getArraySpecifiers();

      // Should be an array
      assert (arrDimSpec != null && !arrDimSpec.isEmpty());

      // Create new array declaration with modified top dimension
      aSpec = new ArraySpecifier();
      int dimNum = ((ArraySpecifier) arrDimSpec.get(0)).getNumDimensions();
      LinkedList<Expression> arrayDims = new LinkedList<Expression>();
      for (int j = 0; j < dimNum; ++j)
        if (j != dimNum - numDims)
          // Add rest of dimensions unaltered to array dim list
          arrayDims.add(((ArraySpecifier) arrDimSpec.get(0)).getDimension(j));
        else
          // Divide dimension corresponding to highest thread Dim (x, y, or z)
          // by mem-partition factor and add to array dim list
          arrayDims.add((Expression) new BinaryExpression(
                ((ArraySpecifier) arrDimSpec.get(0)).getDimension(dimNum - numDims).clone(),
                BinaryOperator.DIVIDE,
                new IntegerLiteral(mempartFactor)));

      System.out.println("Array Specs for " + idexp.toString() + ": "
          + arrDimSpec.toString() + " - " + arrayDims.toString());
      aSpec.setDimensions(arrayDims);
      // Store new partitioned specifier for future use
      mID2ArrSpec.put(idexp, aSpec);
    }
    return aSpec;
  }

  private void mempartDeclStatement(DeclarationStatement dStmt, List<Declaration> partDecls) 
  {
    System.out.println("YOO");
    Declaration decl = dStmt.getDeclaration();
    System.out.println(" --- Handling Declaration: "+decl.toString());
    Declarator declor = ((VariableDeclaration)decl).getDeclarator(0);
    IDExpression ide = declor.getID();

    if (inArrayPartList(ide.toString())) {
      // Create new array declaration with modified top dimension
      LinkedList<Specifier> leadSpecs = new LinkedList<Specifier>();
      LinkedList<Specifier> trailSpecs = new LinkedList<Specifier>();
      LinkedList<Specifier> leadSpecsDecl = new LinkedList<Specifier>();

      trailSpecs.add(partArrDims(declor));
      leadSpecs.addAll(declor.getSpecifiers());
      leadSpecsDecl.addAll(((VariableDeclaration) decl).getSpecifiers());

      // Create partioned array declarations
      for (int i = mempartFactor - 1; i >= 0; i--) {//YOO
        IDExpression newPartId = new NameID(declor.getID().toString()+ Array_Suffix + i);
        VariableDeclarator newPartition = new VariableDeclarator(leadSpecs, newPartId, trailSpecs);
        VariableDeclaration newdecl = new VariableDeclaration(leadSpecsDecl, newPartition);
        System.out.println(" --- Map declaration: "+newdecl.toString()+" to "+newPartition.toString());
        mId2PartArrSym.put(newPartId.toString(), newdecl);
        partDecls.add((Declaration)newdecl);
      }
    }
  }


  private void mempartTaskCall(ExpressionStatement stmt) 
  {
    FunctionCall coreCall = (FunctionCall)(stmt.getExpression());

    System.out.println(" --- Handling FunctionCall: "+coreCall.toString());

    List<Expression> args = coreCall.getArguments();
    List<Expression> newArgs = new LinkedList<Expression>();

    int index = 0;
    int found = 0;
    for (Expression arg : args) {
      if (arg instanceof IDExpression && inArrayPartList(arg.toString())) {
        for (int i = 0 ; i < mempartFactor ; i++) {

          System.out.println(" --- Handling argument: "+arg.toString());

          String tmpStr = arg.toString()+ Array_Suffix + i;
          assert(mId2PartArrSym.containsKey(tmpStr) == true);
          VariableDeclaration argDecl = mId2PartArrSym.get(tmpStr);

          System.out.println(" --- Symbol decl: "+argDecl.toString());

          IDExpression newPartId = new Identifier((VariableDeclarator)argDecl.getDeclarator(0));
          newArgs.add((Expression)newPartId);
        }
        found++;
        FCUDAGlobalData.updateCommonArgsIndex(coreCall, index, (mempartFactor-1)*found);
      } else
        newArgs.add(arg.clone());
      index++;
    }

    coreCall.setArguments(newArgs);

  }

  private void mempartIfStatement(IfStatement stmt) 
  {
    System.out.println("[IfStatement] " + stmt.toString());
    System.out.println("[Control] " + stmt.getControlExpression());
    Expression control = stmt.getControlExpression();
    Set<Expression> useIDs = DataFlow.getUseSet(control);
    for(Expression id : useIDs) {
      if (id instanceof IDExpression) {
        if (mUnrollIDinfo.containsKey((IDExpression) id))
          mUnrollIDinfo.get((IDExpression)id).isCtrlFlow = true;
      }
    }
    mempartCompoundStatement((CompoundStatement) stmt.getThenStatement());
    mempartCompoundStatement((CompoundStatement) stmt.getElseStatement());
    // We need to update the control expression
    // as well if it contains array to be partitioned
    mempartDefaultStatement_non_consecutive(control);
  }

  private void mempartWhileLoopStatement(WhileLoop stmt) 
  {
    System.out.println("[WhileloopStatement] " + stmt.toString());
    mempartCompoundStatement((CompoundStatement) stmt.getBody());
  }

  private void mempartForloopStatement(ForLoop stmt) 
  {
    System.out.println("[ForloopStatement] " + stmt.toString());
    mempartCompoundStatement((CompoundStatement) stmt.getBody());
  }

  private void mempartThreadLoopStatement(Statement stmt) 
  {
    partDims = numDims - 1;
    tloop = (ThreadLoop) stmt;
    isUpdatedTLoop = false;

    loopIndex = MCUDAUtils.Tidx.getId(partDims);
    loopBound = tloop.getBound(partDims);

    // update loop condition
    if ((procEnum & COMP_PROC) > 0) {
      Expression expr = null;
      if (bconsUnroll) {
        System.out.println("[update loop bound/index]");
        expr = (Expression) new BinaryExpression((Expression)loopBound.clone(),
            BinaryOperator.DIVIDE,
            new IntegerLiteral(mempartFactor));
        tloop.setBound(partDims, expr);

        // update loop step
        BinaryExpression bexpr = (BinaryExpression) tloop.getUpdate(partDims);
        int val = (int) ((IntegerLiteral) bexpr.getRHS()).getValue() / mempartFactor;
        Expression step = new IntegerLiteral(val);

        expr = (Expression) new BinaryExpression((Expression)bexpr.getLHS().clone(),
            BinaryOperator.ADD,
            new IntegerLiteral(val));
        tloop.setUpdate(partDims, expr);
      }
    }
    mempartCompoundStatement((CompoundStatement) tloop.getBody());
    if (shapeList != null) {
      int shape = Integer.parseInt(shapeList.get(0));
      if (shape != 0) {
        loopBound = new IntegerLiteral(shape);
        BinaryExpression newLoopBound = new BinaryExpression(new IntegerLiteral(shape),
            BinaryOperator.DIVIDE,
            new IntegerLiteral(mempartFactor));
        tloop.setBound(partDims, newLoopBound);
      }
    }
  }

  private void mempartCompoundStatement(CompoundStatement cStmt) 
  {
    System.out.println("HOO " + cStmt);
    if (cStmt == null)
      return;

    Map<Statement, List<Declaration>> mStmt2partDecls = new HashMap<Statement, List<Declaration>>();
    FlatIterator flatIter = new FlatIterator(cStmt);

    while (flatIter.hasNext()) {
      Object currObj = flatIter.next();
      if (!(currObj instanceof Statement)) {
        System.out.println("Child " + currObj.toString()
            + " of compound statement is not statement");
        System.exit(0);
      }
      Statement currStmt = (Statement) currObj;

      if (currStmt instanceof ThreadLoop) {
        mempartThreadLoopStatement(currStmt);
        continue;
      }
      System.out.println("HEY " + currStmt + " " + getStmtTypeId(currStmt));
      switch (getStmtTypeId(currStmt)) {
        case DeclarationStatement:
          if ((procEnum & KERN_PROC) > 0) {
            List<Declaration> partDecls = new LinkedList<Declaration>();
            mempartDeclStatement((DeclarationStatement)currStmt, partDecls);
            if (!partDecls.isEmpty())
              mStmt2partDecls.put(currStmt, partDecls);
          }
          break;
        case ForLoop:
          mempartForloopStatement((ForLoop) currStmt);
          break;
        case IfStatement:
          mempartIfStatement((IfStatement) currStmt);
          break;
        case CompoundStatement:
          mempartCompoundStatement((CompoundStatement) currStmt);
          break;
        case WhileLoop:
          mempartWhileLoopStatement((WhileLoop) currStmt);
          break;
        case ExpressionStatement:
          if ((procEnum & KERN_PROC) > 0) {
            if (mCoreStmts.contains(currStmt))
              mempartTaskCall((ExpressionStatement)currStmt);
          } else
            mempartDefaultStatement_wrap(currStmt);
          break;
        default:
          mempartDefaultStatement_wrap(currStmt);
          break;
      }
    }

    // Clean old declaration statements
    for (Statement dStmt : mStmt2partDecls.keySet()) {
      for (Declaration newDcl : mStmt2partDecls.get(dStmt))
        cStmt.addDeclarationBefore(((DeclarationStatement)dStmt).getDeclaration(), newDcl);
      cStmt.removeChild((Traversable)dStmt);
    }
    mStmt2partDecls.clear();
  }

  public void updateProcParameter(Procedure proc) 
  {
    List<Declaration> oldDeclLst = new LinkedList<Declaration>();
    for (Object declObj : proc.getParameters()) {
      if (!(declObj instanceof VariableDeclaration)) {
        System.out.println("Parameter:" + declObj.toString() + " of Procedure: " +
            proc.getName().toString() + "is not a VariableDeclaration");
        System.exit(0);
      }

      // *AP*
      // We have created these declarations, so we know they
      // are VariableDeclarations with a single Declarator
      VariableDeclaration pDecl = (VariableDeclaration)declObj;
      Declarator pDeclor = pDecl.getDeclarator(0);
      IDExpression ide = pDeclor.getID();

      if (inArrayPartList(ide.toString())) {
        List<Specifier> arrDimSpec = pDeclor.getArraySpecifiers();

        // Should be an array
        assert (arrDimSpec != null && !arrDimSpec.isEmpty());

        // Create new array declaration with modified top dimension
        LinkedList<Specifier> leadSpecs = new LinkedList<Specifier>();
        LinkedList<Specifier> trailSpecs = new LinkedList<Specifier>();
        LinkedList<Specifier> leadSpecsDecl = new LinkedList<Specifier>();
        ArraySpecifier aSpec = new ArraySpecifier();
        int dimNum = ((ArraySpecifier) arrDimSpec.get(0)).getNumDimensions();
        LinkedList<Expression> arrayDims = new LinkedList<Expression>();

        for (int j = 0; j < dimNum; ++j)
          if (j != dimNum - numDims)
            // Add rest of dimensions unaltered to array dim list
            arrayDims.add(((ArraySpecifier) arrDimSpec.get(0)).getDimension(j));
          else
            // Divide dimension corresponding to highest thread Dim (x, y, or z)
            // by mem-partition factor and add to array dim list
            arrayDims.add((Expression) new BinaryExpression(
                  ((ArraySpecifier) arrDimSpec.get(0)).getDimension(dimNum - numDims).clone(),
                  BinaryOperator.DIVIDE,
                  new IntegerLiteral(mempartFactor)));

        System.out.println("Array Specs for " + ide.toString() + ": "
            + arrDimSpec.toString() + " - " + arrayDims.toString());

        aSpec.setDimensions(arrayDims);
        trailSpecs.add(aSpec);
        leadSpecs.addAll(pDeclor.getSpecifiers());
        leadSpecsDecl.addAll(((VariableDeclaration) pDecl)
            .getSpecifiers());

        // Replicate old array declaration
        for (int i = mempartFactor - 1; i >= 0; i--) {
          IDExpression newPartId = new NameID(pDeclor.getID().toString()+ Array_Suffix + i);
          VariableDeclarator newPartition = new VariableDeclarator(leadSpecs,
              newPartId,
              trailSpecs);
          VariableDeclaration newdecl = new VariableDeclaration(leadSpecsDecl, newPartition);
          proc.addDeclarationAfter(pDecl, (Declaration) newdecl);

          // Update FCUDAGlobalData
          FCUDAGlobalData.partitionBRAM(ide, newPartId);
        }
        oldDeclLst.add(pDecl);
      }
    }

    // Remove old array declaration
    for (Declaration oldDecl : oldDeclLst)
      proc.removeDeclaration(oldDecl);

  } // updateProcParameter()



  private void handleThrDefIDs()
  {
    // Replicate unrolled variable definitions if used in conditional statements
    for (IDExpression uID : mUnrollIDinfo.keySet()) {
      UnrollIDinfo uInfo = mUnrollIDinfo.get(uID);

      System.out.println(" *** handleThrDef() for "+uID.toString());
      System.out.println(" uses: "+uInfo.useStmt.size()+"  control flow var: "+uInfo.isCtrlFlow);

      if (!(uInfo.useStmt.isEmpty()) && (uInfo.isCtrlFlow == true)) {
        // Handle definition statement
        CompoundStatement defParent = (CompoundStatement) Tools.getAncestorOfType(uInfo.defStmt, CompoundStatement.class);
        VariableDeclaration uIDdecl = FCUDAutils.getVariableDeclaration(uID);
        CompoundStatement declParent = (CompoundStatement) Tools.getAncestorOfType(uIDdecl, CompoundStatement.class);
        VariableDeclarator uIDdeclor = FCUDAutils.getDeclaratorFromDeclaration(uIDdecl, uID);
        NameID newUid = new NameID(uID.toString()+"_mp");
        VariableDeclarator newUidDeclor = new VariableDeclarator(uIDdeclor.getSpecifiers(),
            newUid,
            uIDdeclor.getTrailingSpecifiers());
        VariableDeclaration newUidDecl = new VariableDeclaration(uIDdecl.getSpecifiers(),
            newUidDeclor);
        NameID oldUidx = new NameID(idxName+uInfo.old_step);
        NameID newUidx = new NameID(idxName+uInfo.new_step);
        Statement defCopy = (Statement) uInfo.defStmt.clone();
        IRTools.replaceAll(defCopy, uID, new Identifier(newUidDeclor)); // replace defined variable
        IRTools.replaceAll(defCopy, oldUidx, newUidx); // replace unroll index variable
        defParent.addStatementAfter(uInfo.defStmt, defCopy);
        declParent.addDeclarationAfter(uIDdecl, newUidDecl);

        // Handle maydef statements
        for (Statement mdStmt : uInfo.mayDefStmt) {
          CompoundStatement maydefParent = (CompoundStatement) Tools.getAncestorOfType(mdStmt, CompoundStatement.class);
          Statement mdCopy = (Statement) mdStmt.clone();
          IRTools.replaceAll(mdCopy, uID, new Identifier(newUidDeclor));
          maydefParent.addStatementAfter(mdStmt, mdCopy);
        }

        // Handle use statements
        for (Statement usestmt : uInfo.useStmt)
          IRTools.replaceAll(usestmt, uID, new Identifier(newUidDeclor));
      }
    }
  }

  public void partition(Procedure proc) 
  {
    numDims = FCUDAGlobalData.getKernTblkDim(mProcedure);
    System.out.println("[numDims]" + numDims);

    // Get memory partition factor
    System.out.println("[Memory partition] : " + proc.getName() + "\n");
    System.out.println("[Proc]: " + proc.toString() + "\n");
    mempartFactor = FCUDAutils.getTaskMempart(proc);
    shapeList = FCUDAutils.getTaskShape(proc);
    System.out.println("[mempartFactor]" + mempartFactor);
    // Clear Unroll ID info
    mUnrollIDinfo.clear();

    if (mempartFactor > 1) {
      mPartitionArrays = FCUDAutils.getTaskSplitArray(proc);
      System.out.println("[Arrays]" + mPartitionArrays.toString());

      /*
       * fixme: assumption: we assume the memory partition for the arrays
       * which are unrolled before.
       */
      // findMaxStep(proc);

      maxStep = FCUDAutils.getTaskUnroll(proc);
      sizeMempart = maxStep / mempartFactor;
      System.out.println("[Max Step] ------ " + maxStep);
      System.out.println("[sizeMem ] ------ " + sizeMempart);
      updateProcParameter(proc);

      if (FCUDAutils.getTaskType(proc).equals("compute"))
        procEnum = COMP_PROC;
      else {
        String name = proc.getName().toString();
        int lastIndex = name.lastIndexOf("fetch");
        if (lastIndex < 0)
          procEnum = TRANS_WRITE_PROC;
        else
          procEnum = TRANS_FETCH_PROC;
      }
      mempartCompoundStatement(proc.getBody());
      handleThrDefIDs();
    }
  }

  public void transformProcedure(Procedure proc) 
  {
    mProcedure = proc;
    mID2ArrSpec.clear(); // clear old info before moving to next kernel
    bconsUnroll = false;
    List<Procedure> tskLst = FCUDAutils.getTaskMapping(proc.getSymbolName());
    for (Procedure task : tskLst) {
      tloop = null;
      loopBound = null;
      partition(task);
    }

    // Memory partition in kernel function
    System.out.println("[Memory partition] : " + proc.getName() + "\n");
    List<FunctionCall> fcudaCores = FCUDAGlobalData.getFcudaCores(proc);
    for (FunctionCall currCore : fcudaCores) {
      mCoreStmts.add(FCUDAutils.getClosestParentStmt(currCore));
    }

    procEnum = KERN_PROC;
    System.out.println("HAA " + mempartFactor + " " + proc.getBody());
    if (mempartFactor > 1) {
      mempartCompoundStatement(proc.getBody());
      // Remove old BRAMS in FCUDAGlobalData
      //for (String arrStr : mPartitionArrays) {
        //NameID arrID = new NameID(arrStr);
        //FCUDAGlobalData.removeBRAM(arrID);
      //}
    }
  }
}
