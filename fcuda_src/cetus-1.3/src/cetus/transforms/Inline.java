package cetus.transforms;

import cetus.hir.*;
import cetus.analysis.*;
import java.util.*;

/**
 * Searches for small functions that can be inlined
 * and inlines their code into the call site.
 */
public class Inline
{
  private static String pass_name = "[Inline]";

  /**
   * John A. Stratton
   * For derivative classes, it makes sense to have the program be protected
   */
  protected Program program;
  private int threshold;

  public Inline(Program program, int threshold)
  {
    this.program = program;
    this.threshold = threshold;
  }

  private HashSet<String> findInlineCandidates()
  {
    HashSet<String> candidates = new HashSet<String>();

    BreadthFirstIterator iter = new BreadthFirstIterator(program);
    iter.pruneOn(Procedure.class);

    for (;;)
    {
      Procedure proc = null;

      try {
	  proc = (Procedure)iter.next(Procedure.class);
      } catch (NoSuchElementException e) {
        break;
      }

      if (proc.getBody().countStatements() < threshold)
        candidates.add(proc.getName().toString());
    }

    return candidates;
  }

  protected void inlineCall(FunctionCall call)
  {
    //Not handled yet
    if(call instanceof KernelLaunch)
      throw new IllegalArgumentException();

    /* get the Procedure object for the call */
    Procedure proc = call.getProcedure();

    /* check that the number of arguments == number of params */
    if (call.getNumArguments() != proc.getNumParameters())
    {
      System.err.println(pass_name + " call to procedure " + proc.getName().toString()
        + " appears to have an incorrect number of arguments");
      return;
    }

    /* clone the body of the Procedure object */
    CompoundStatement inlined_code = (CompoundStatement)proc.getBody().clone();

    /* before we go any further, make sure there's actually something in the 
       function to inline */
    if(inlined_code.countStatements() == 0)
      return;

    /* Get the call site and the compound statement enclosing it */
    Traversable t = call;
    while (!(t instanceof Statement))
      t = t.getParent();
    Statement stmt = (Statement)t;
    CompoundStatement context = (CompoundStatement)stmt.getParent();

    /* iterate over inline_code, find a use of a parameter, and replace
       it with its corresponding argument */
    List<Declaration> params = proc.getParameters();
    List<Expression> args = call.getArguments();
    int n = call.getNumArguments();
    Set<Expression> function_defs = DataFlow.defines(inlined_code);
    function_defs.addAll(DataFlow.mayDefine(inlined_code));

    context.addStatementBefore(stmt, inlined_code);

    for (int i = 0; i < n; ++i)
    {
      List<Specifier> paramiType = ((VariableDeclaration)params.get(i)).getSpecifiers();
//*AP* cetus-1.3     Identifier parami = (Identifier)((VariableDeclaration)params.get(i)).getDeclaredSymbols().get(0);
      Identifier parami = (Identifier)((VariableDeclaration)params.get(i)).getDeclaredIDs().get(0);
      /* if the parameter is never actually written to, it's safe to do a simple find-and-replace */
      if(!function_defs.contains(parami))
        Tools.replaceAll(inlined_code, parami, args.get(i));
      else 
      {
        /* add a new variable to copy the argument value to, 
         * and make sure the rest of the function uses it */
        Identifier newParam;
        /* Easy case, we can use the original parameter name */
        if(context.findSymbol(parami) == null)
	  newParam = parami;
        else
          /* TODO: what we really need is a temp that doesn't collide with any symbol visible from 
           * any location the parameter was used in the inlined code */
          newParam = Tools.getTemp(inlined_code, parami);
        
        DeclarationStatement newParamStmt = new DeclarationStatement(new VariableDeclaration(paramiType, 
                                                new VariableDeclarator(newParam) )  );
        /* Checked for empty body at beginning, must be at least one child */
        inlined_code.addStatementBefore((Statement)inlined_code.getChildren().get(0), newParamStmt);
        Statement paramAssignment = new ExpressionStatement(new AssignmentExpression(newParam, 
						    AssignmentOperator.NORMAL, args.get(i) ) );
        inlined_code.addStatementAfter(newParamStmt, paramAssignment);

        // If we had to rename to parameter to prevent name aliasing, search and replace in the body
        if(newParam.compareTo(parami) != 0)
          Tools.replaceAll(inlined_code, parami, newParam);
      }

    } //for all args

    //JAS: Need to support unary expressions as well
    if (call.getParent() instanceof BinaryExpression
        || call.getParent() instanceof FunctionCall
        || call.getParent() instanceof UnaryExpression)
    {
      /* Case 1: x = foo(); x = foo() + y; bar(foo()); */

      /* Get the return statement of the function */
      DepthFirstIterator iter = new DepthFirstIterator(inlined_code);
      iter.pruneOn(Expression.class);
      ReturnStatement func_stmt; 
      try { 
	  func_stmt = (ReturnStatement)iter.next(ReturnStatement.class);
      } catch (NoSuchElementException e) {
        /* Something's not consistent, if the function is used in an expression, it 
         * really should have returned something */
        throw new InternalError();
      }     

      /* TODO: make sure that the single return pass has been run so that
         there is only one return and there is a return variable */
      Expression ret_var = func_stmt.getExpression();

      /* get the name of the return variable and replace the call
         with a copy */

      /* TODO: move the declaration of the return variable to before the
         inlined_code */
      
      if(ret_var instanceof Identifier){
        Identifier ret_name = (Identifier)ret_var;
        Identifier temp = Tools.getTemp(inlined_code, ret_name);
        Declaration decl = Tools.findSymbol(inlined_code, ret_name);
        DeclarationStatement decl_stmt = (DeclarationStatement)decl.getParent();
        decl_stmt.detach();
        decl = Tools.findSymbol(inlined_code, temp);
        decl_stmt = (DeclarationStatement)decl.getParent();
      /* find the declaration of the return variable */
        Tools.replaceAll(inlined_code, ret_name, temp); 
        decl_stmt.detach();
        context.addDeclaration(decl);
        ret_var = temp;
      }

      call.swapWith((Expression)ret_var.clone());
    }
    else
    {
      /* Case 2: foo(); */

      /* eliminate the call */
      stmt.detach();
    }

    /* eliminate any ReturnStatements from inlined_code */
    DepthFirstIterator inline_iter = new DepthFirstIterator(inlined_code);
    inline_iter.pruneOn(Expression.class);

    for (;;)
    {
      ReturnStatement ret = null;

      try {
	ret = (ReturnStatement)inline_iter.next(ReturnStatement.class);
      } catch (NoSuchElementException e) {
        break;
      }

      ret.detach();
    }
  }

  public static void run(Program program, int threshold)
  {
    Tools.printlnStatus(pass_name + " begin", 1);
    (new Inline(program, threshold)).start();
    Tools.printlnStatus(pass_name + " end", 1);
  }

  public void start()
  {
    HashSet<String> candidates = findInlineCandidates();

    DepthFirstIterator iter = new DepthFirstIterator(program);

    for (;;)
    {
      FunctionCall call = null;

      try {
	call = (FunctionCall)iter.next(FunctionCall.class);
      } catch (NoSuchElementException e) {
        break;
      }

      if (candidates.contains(call.getName().toString()))
        inlineCall(call);
    }
  }
}
