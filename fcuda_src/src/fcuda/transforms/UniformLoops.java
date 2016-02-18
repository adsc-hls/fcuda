package fcuda.transforms;
import fcuda.*;
import fcuda.common.*;

import cetus.hir.*;
import cetus.analysis.*;  //*AP*

import java.util.*;

/**
 * Changes all loops so that they have a single value/variable as their 
 * condition.  This requires that for loops be changed into while loops, 
 * so that initialization and updates can be done outside of the loop 
 * declaration.
 */
public class UniformLoops extends KernelTransformPass
{

  public UniformLoops(Program program)
  {
    super(program);
  }

  public String getPassName()
  {
    return new String("[UniformLoops]");
  }

  public List<Statement> statementsFromCommaExpression(CommaExpression e)
  {
    List<Statement> stmts = new LinkedList<Statement>();
    for(Traversable t : e.getChildren() )
      stmts.add(new ExpressionStatement(((Expression)t).clone()));
    return stmts;
  }

  public void transformLoop(Loop loop)
  {
    //All loops implement Statement and SymbolTable, but 
    //we can't access them directly that way, so we cast.
    CompoundStatement loopContext = (CompoundStatement)loop.getParent();
    CompoundStatement body = (CompoundStatement)loop.getBody();
    SymbolTable loopsym = (SymbolTable)loop;
    Statement loopStmt = (Statement)loop;
    Expression cond = loop.getCondition();
    //*cetus-1.1*   Identifier newCondition = null;
    IDExpression newCondition = null;


    if (loop instanceof ForLoop)
    {
      //Convert to a while loop
      ForLoop forloop = (ForLoop)loop;
      Statement init = forloop.getInitialStatement();
      forloop.setInitialStatement(null);
      if (init instanceof ExpressionStatement && 
          ((ExpressionStatement)init).getExpression() instanceof CommaExpression)
      {
        CommaExpression ce = (CommaExpression)((ExpressionStatement)init).getExpression();
        for(Statement s : statementsFromCommaExpression(ce) )
          loopContext.addStatementBefore(forloop, s);
      }
      else 
        loopContext.addStatementBefore(forloop, init);

      forloop.setBody(null);
      Expression update = forloop.getStep();

      forloop.setStep(null);
      if (update instanceof CommaExpression)
      {
        for(Statement s : statementsFromCommaExpression((CommaExpression)update))
          body.addStatement(s);
      }
      else 
        body.addStatement(new ExpressionStatement(update));

      WhileLoop newloop = new WhileLoop((Expression)cond.clone(), body);

      newloop.swapWith(forloop);

      loop = newloop;
      loopStmt = newloop;
    }

    DepthFirstIterator i = new DepthFirstIterator(cond);
    FunctionCall f = null;
    try{
      f = (FunctionCall)i.next(FunctionCall.class);
    } catch (NoSuchElementException e) {}
    if(f != null || DataFlow.defines(cond).size() > 0) {
      newCondition = SymbolTools.getUnusedID(loopsym);
      AssignmentExpression evaluate = 
        new AssignmentExpression((IDExpression)newCondition, 
            AssignmentOperator.NORMAL, 
            (Expression)cond.clone() );
      ExpressionStatement evalStmt = new ExpressionStatement(evaluate);

      body.addStatement(evalStmt);

      loop.setCondition((Expression)newCondition.clone());

      //Add the declaration to the parent
      VariableDeclarator condDecl = 
        new VariableDeclarator(newCondition.clone());

      VariableDeclaration condDeclaration = 
        new VariableDeclaration(Specifier.INT, condDecl);

      loopContext.addDeclaration(condDeclaration);

      //Need to add a precondition evaluation if it's a while loop
      if(loop instanceof WhileLoop) {
        loopContext.addStatementBefore(loopStmt, (Statement)evalStmt.clone());
      }
    }
  }

  public void transformProcedure(Procedure proc)
  {
    BreadthFirstIterator iter = new BreadthFirstIterator(proc);

    for (;;)
    {
      Loop loop = null;

      try {
        loop = (Loop)iter.next(Loop.class);
      } catch (NoSuchElementException e) {
        break;
      }

      transformLoop(loop);
    }
  }
}
