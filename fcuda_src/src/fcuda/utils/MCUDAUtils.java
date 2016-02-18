package fcuda.utils;

import java.util.*;

import cetus.exec.*;
import cetus.hir.*;

import fcuda.ir.*;

/* Utility functions for MCUDA transformations */

public class MCUDAUtils
{
  public static final Dim3Var Bdim = new Dim3Var("blockDim", 3);
  public static final Dim3Var Tidx = new Dim3Var("threadIdx", 3);
  public static final Dim3Var Bidx = new Dim3Var("blockIdx", 3);
  public static final Dim3Var Gdim = new Dim3Var("gridDim", 3);
  public static final Dim3Var StartIdx = new Dim3Var("__startIdx", 3);
  public static final Dim3Var EndIdx = new Dim3Var("__endIdx", 3);
  public static final ImplicitVariable LocalTidx = new ImplicitVariable("__threadIndex", Specifier.INT);

  public static String getParamsString(String funcName)
  {
    return new String(funcName + "_params");
  }

  public static NameID getParamsID(String funcName)
  {
    return new NameID(getParamsString(funcName));
  }

  public static Expression getMaxNumThreads()
  {
    int tpb;
    if(Driver.getOptionValue("tpb") != null)
      tpb = Integer.valueOf(Driver.getOptionValue("tpb"));
    else
      tpb = 512;

    if(Driver.getOptionValue("numSharingThreads") != null) {
      int threads = Integer.valueOf(Driver.getOptionValue("numSharingThreads"));
      return new IntegerLiteral( (tpb + threads - 1) / threads );
    }
    else
      return new IntegerLiteral(tpb);
  }

  public static int getBdim(int index)
  {
    String size = Driver.getOptionValue("dimb" + Bdim.getDimEntry(index));
    if(size == null)
      throw new InternalError("Trying to get unknown Bdim(" + index + ")");

    return Integer.parseInt(size);
  }

  public static List<Expression> getBdim()
  {
    if(Driver.getOptionValue("specifyDims") == null)
      throw new InternalError("Trying to get unknown Bdim");

    List<Expression> dimList = new LinkedList<Expression>();

    for(int i = 0; i < Bdim.getNumEntries(); i++) {
      int size = getBdim(i);
      if(size > 1)  dimList.add(new IntegerLiteral(size));
    }
    return dimList;
  }

  public static AccessOperator paramsAccessOperator()
  {
    if(Driver.getOptionValue("staticKernelParams") == null)
      return AccessOperator.POINTER_ACCESS;
    else
      return AccessOperator.MEMBER_ACCESS;
  }

  public static List<Declaration> getKernelParams(String funcName)
  {
    List<Declaration> paramList = new LinkedList<Declaration>();

    if (Driver.getOptionValue("staticKernelParams") == null &&
        Driver.getOptionValue("packedKernelParams") != null ) {
      List<Specifier> paramsType = new ChainedList<Specifier>();
      paramsType.add(Specifier.CONST);
      paramsType.add(Specifier.VOID);
      paramsType.add(PointerSpecifier.UNQUALIFIED);
      VariableDeclaration paramDecl;
      VariableDeclarator params = new VariableDeclarator(getParamsID(funcName));
      paramDecl = new VariableDeclaration(paramsType, params);
      paramList.add(paramDecl);
    }

    paramList.addAll(Bidx.getDecl());
    paramList.addAll(Bdim.getDecl());
    paramList.addAll(Gdim.getDecl());
    if (Driver.getOptionValue("staticTasks") != null) {
      paramList.addAll(StartIdx.getDecl());
      paramList.addAll(EndIdx.getDecl());
    }

    return paramList;
  }

  public static FunctionCall getSync()
  {
    NameID syncName = new NameID("__syncthreads");

    FunctionCall syncCall = new FunctionCall(syncName);

    return syncCall;
  }

  public static boolean isThreadLoopBody(CompoundStatement stmt)
  {
    return (stmt.getParent() instanceof ThreadLoop);
  }

  public static boolean isThreadLoop(Loop loop)
  {
    return (loop instanceof ThreadLoop);
  }

  public static CompoundStatement getThreadLoopBody(Traversable node)
  {
    while (node.getParent() != null) {
      node = node.getParent();
      if (node instanceof CompoundStatement) {
        if (isThreadLoopBody((CompoundStatement)node)) {
          return (CompoundStatement)node;
        }
      }
    }

    return null;
  }

  public static ThreadLoop getOuterThreadLoop(Traversable node)
  {
    while(node.getParent() != null) {
      node = node.getParent();
      if(node instanceof ThreadLoop) {
        return (ThreadLoop)node;
      }
    }

    return null;
  }

  public static Set<CompoundStatement> getThreadLoops(Traversable root)
  {
    Set<CompoundStatement> blocks = new HashSet<CompoundStatement>();

    DepthFirstIterator iter = new DepthFirstIterator(root);
    iter.pruneOn(Expression.class);

    while(iter.hasNext()) {
      ThreadLoop tloop = null;
      try{
        tloop = (ThreadLoop)iter.next(ThreadLoop.class);
      } catch (NoSuchElementException e) {
        break;
      }
      blocks.add((CompoundStatement)tloop.getBody());
    }

    return blocks;
  }

  public static Expression getSMID()
  {
    return new FunctionCall(new NameID("getSMID"));
  }

  @SuppressWarnings("unused")
  public static ThreadLoop NewNestedThreadLoop(int threadDim,
      CompoundStatement body)
  {
    return new ThreadLoop(body,threadDim);
  }

  public static void addStructTypedefBefore(TranslationUnit tu, Declaration d, ClassDeclaration struct)
  {
    IDExpression structID = struct.getName();
    // Create the typedef for terser access
    List<Specifier> typeList = new LinkedList<Specifier>();
    typeList.add(Specifier.TYPEDEF);
    //Shouldn't need a user specifier for this, but it looks like
    //there isn't another way.
    //typeList.add(new UserSpecifier(new Identifier("struct")));

    UserSpecifier structType = new UserSpecifier(new NameID("struct "+structID.toString()));
    typeList.add(structType);
    VariableDeclarator typedef =
      new VariableDeclarator((Identifier)structID.clone());

    VariableDeclaration typeDecl =
      new VariableDeclaration(typeList, typedef);

    tu.addDeclarationBefore(d, typeDecl);
  }

  public static List<DeclarationStatement> getScopeDecls(CompoundStatement scope)
  {
    LinkedList<DeclarationStatement> statements = new LinkedList<DeclarationStatement>();

    for (Traversable t : scope.getChildren()) {
      if (t instanceof DeclarationStatement)
        statements.add((DeclarationStatement)t);
    }
    return statements;
  }

  public static Traversable
    getContextOfInterest(Statement stmt,
        Set<Class<? extends Traversable>> of_interest)
  {
    Traversable a = stmt;
    do {
      a = a.getParent();
    }while (! of_interest.contains(a.getClass()));
    return a;
  }

  public static CompoundStatement getContinueContext(ContinueStatement stmt)
  {
    Set<Class<? extends Traversable>> of_interest =
      new HashSet<Class<? extends Traversable>>();
    of_interest.add(ForLoop.class);
    of_interest.add(WhileLoop.class);
    of_interest.add(DoLoop.class);

    return (CompoundStatement)
      ((Loop)getContextOfInterest(stmt, of_interest)).getBody();
  }

  public static CompoundStatement getBreakContext(BreakStatement stmt)
  {
    Set<Class<? extends Traversable>> of_interest =
      new HashSet<Class<? extends Traversable>>();
    of_interest.add(ForLoop.class);
    of_interest.add(WhileLoop.class);
    of_interest.add(DoLoop.class);
    of_interest.add(SwitchStatement.class);

    Traversable a = getContextOfInterest(stmt, of_interest);
    if (a instanceof Loop)
      return (CompoundStatement)((Loop)a).getBody();
    else
      return ((SwitchStatement)a).getBody();
  }

  public static CompoundStatement getCaseContext(Case stmt)
  {
    Set<Class<? extends Traversable>> of_interest =
      new HashSet<Class<? extends Traversable>>();
    of_interest.add(SwitchStatement.class);

    return ((SwitchStatement)getContextOfInterest(stmt, of_interest)).getBody();
  }

  //This function swaps each instance of the list of input expressions
  // with an instance of the first provided expression if the original
  // expression is inside of a ThreadLoop, and with the second expression
  // if the original expression was outside of a ThreadLoop.
  public static void swapWithExprs(List<Expression> references,
      Expression inside_tLoop_expr, Expression outside_tLoop_expr)
  {
    for (Expression use : references) {
      //If it's an access outside of the thread loops, don't
      //buffer it.  Instead, just access element 0
      if (getThreadLoopBody(use) != null)
        use.swapWith((Expression)inside_tLoop_expr.clone());
      else
        use.swapWith((Expression)outside_tLoop_expr.clone());
    }
  }

  public static List<Expression> getInstances(VariableDeclaration vDecl,
      Traversable root)
  {
    PostOrderIterator iter = new PostOrderIterator(root);
    LinkedList<Expression> uses = new LinkedList<Expression>();

    while (iter.hasNext()) {
      Identifier variableID = null;

      try {
        variableID = (Identifier)iter.next(Identifier.class);
      } catch (NoSuchElementException e) {
        break;
      }

      //If we've just found a declaration or shadow reference, don't include it
      if (!(variableID.getParent() instanceof Declarator) &&
          variableID.findDeclaration() == vDecl) {
        //And, if it's a field of another variable, it doesn't count
        if (!(variableID.getParent() instanceof AccessExpression &&
              ((BinaryExpression)variableID.getParent()).getRHS() == variableID ))
          uses.add(variableID);
      }
    }
    return uses;
  }

  public static List<VariableDeclaration> getBdimDecl()
  {
    return MCUDAUtils.Bdim.getDecl();
  }

  public static List<VariableDeclaration> getGdimDecl()
  {
    return MCUDAUtils.Gdim.getDecl();
  }

  public static Expression getGdimID(int index)
  {
    return MCUDAUtils.Gdim.getId(index);
  }

  public static Expression getBidxID(int index)
  {
    return MCUDAUtils.Bidx.getId(index);
  }

  public static List<VariableDeclaration> getBidxDecl()
  {
    return MCUDAUtils.Bidx.getDecl();
  }

  public static Expression getTidID(int idx)
  {
    return MCUDAUtils.Tidx.getId(idx);
  }

  public static Expression getBdimID(int idx)
  {
    return MCUDAUtils.Bdim.getId(idx);
  }

  public static List<Expression> getBidxID()
  {
    return MCUDAUtils.Bidx.getId();
  }
}
