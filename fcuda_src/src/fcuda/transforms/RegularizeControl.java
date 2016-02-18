package fcuda.transforms;
import fcuda.utils.*;

import cetus.hir.*;
import java.util.*;

public class RegularizeControl extends KernelTransformPass
{
  private static String pass_name = "[RegularizeControl]";

  public String getPassName()
  {
    return new String(pass_name);
  }

  public RegularizeControl(Program program)
  {
    super(program);
  }

  // This function handles control flow representing a jump to the end of a 
  // particular CompoundStatement.  It is applicable for continue statements 
  // for loops, and return statements for function bodies.  
  // The return value is a list of all 
  private void 
    doProcedure(Procedure proc, List<Statement> returns)
    {
      CompoundStatement body = proc.getBody();
      List<Specifier> retType = proc.getReturnType();
      retType.remove(Specifier.GLOBAL);
      retType.remove(Specifier.HOST);
      retType.remove(Specifier.DEVICE);
      Identifier newRetval = null;
      //If the function has a return type, we'll need a return variable
      if(retType.size() > 1 || retType.get(0) != Specifier.VOID)
        newRetval = Tools.getTemp(body, retType, "retval");

      for( Statement retstmt : returns )
      {
        //Relying on IR property that every non CompoundStatement within 
        // a function is the child of a CompoundStatement
        CompoundStatement context = (CompoundStatement)retstmt.getParent();
        List<Traversable> stmtlist = context.getChildren();

        //First of all, any following statements in the direct parent are dead, 
        // if there is no label to reach them by
        List<Traversable> following = new LinkedList<Traversable>();
        following.addAll(
            stmtlist.subList(stmtlist.indexOf(retstmt)+1, stmtlist.size()) );
        boolean isdead = true;
        for(Traversable s : following)
          if(s instanceof Label)
            isdead = false;
        if(isdead)
        {
          stmtlist.removeAll(following);
          for(Traversable t : following)
            t.setParent(null);
        }

        //Next, if the return was in the immediate scope, all is well
        // Otherwise, restructuring is necessary.  

        if(context != body)
        {
          Statement control = (Statement)context.getParent();
          List<Traversable> procStmts = body.getChildren();
          // Can only currently handle the simplest case: early exit
          // at the top level
          if(control instanceof IfStatement && 
              procStmts.contains(control) )
          {
            IfStatement ifStmt = (IfStatement)control;
            List<Traversable> restOfFunction = new LinkedList<Traversable>();
            restOfFunction.addAll(
                procStmts.subList(procStmts.indexOf(control)+1, procStmts.size()) );
            CompoundStatement appendTo = null;
            if(ifStmt.getThenStatement() == context)
            {
              if(ifStmt.getElseStatement() == null)
                ifStmt.setElseStatement(new CompoundStatement());

              appendTo = (CompoundStatement)ifStmt.getElseStatement();
            }
            else // context == ifStmt.getElseStatement()
              appendTo = (CompoundStatement)ifStmt.getThenStatement();

            for(Traversable t : restOfFunction)
              body.removeChild(t);
            for(Traversable t : restOfFunction)
              appendTo.addStatement((Statement)t);


            // The rest of the function may have had other early exits.  
            //  If so, it is safe to treat this Compound Statement as the 
            //  new body to the end of which a return jumps, since the 
            //  fallthrough is the end of the function itself due to the 
            //  previous transformations, and all further returns, due to 
            //  processing order, will be within it.
            body = appendTo; 
            if(newRetval != null)
            {
              Expression retExpr = (Expression)
                ((ReturnStatement)retstmt).getExpression().clone();
              AssignmentExpression assignment = 
                new AssignmentExpression((Identifier)newRetval.clone(), 
                    AssignmentOperator.NORMAL, 
                    retExpr);
              ExpressionStatement record = new ExpressionStatement(assignment);
              context.addStatementBefore(retstmt, record);
            }
            context.removeChild(retstmt);

          }
          else
            throw new InternalError("Can't figure out how to restructure return");
        }
        else
        {
          if(newRetval != null)
          {
            Expression retExpr = (Expression)
              ((ReturnStatement)retstmt).getExpression().clone();
            AssignmentExpression assignment = 
              new AssignmentExpression((Identifier)newRetval.clone(), 
                  AssignmentOperator.NORMAL, 
                  retExpr);
            ExpressionStatement record = new ExpressionStatement(assignment);
            context.addStatementBefore(retstmt, record);
          }
          context.removeChild(retstmt);
        }
      }
      if(newRetval != null)
      {
        ReturnStatement ret = new ReturnStatement((Expression)newRetval.clone());
        proc.getBody().addStatement(ret);
      }
    }

  public void transformProcedure(Procedure proc)
  {

    Set<Class<? extends Traversable>> irregControl = 
      new HashSet<Class<? extends Traversable>>();
    Map<CompoundStatement, List<Statement>> contextToStmtsMap = 
      new HashMap<CompoundStatement, List<Statement>>();

    irregControl.add(GotoStatement.class);
    irregControl.add(Case.class);
    irregControl.add(ReturnStatement.class);
    irregControl.add(ContinueStatement.class);
    irregControl.add(BreakStatement.class);

    DepthFirstIterator iter = new DepthFirstIterator(proc);

    while(iter.hasNext())
    {
      CompoundStatement contextOfInterest = null;
      Statement nextstmt = null;

      //*cetus-1.1*  try{ nextstmt = iter.next(Statement.class); }
      try{ nextstmt = (Statement)iter.next(Statement.class); }
      catch (NoSuchElementException e) { break; }

      if(nextstmt instanceof GotoStatement)
        continue; 
      //Don't actually know what to do here: doGoto((GotoStatement)nextstmt);
      else if (nextstmt instanceof ReturnStatement)
        contextOfInterest = proc.getBody();
      else if (nextstmt instanceof ContinueStatement) {
        ContinueStatement continueStmt = (ContinueStatement)nextstmt;
        contextOfInterest = MCUDAUtils.getContinueContext(continueStmt);
      }else if (nextstmt instanceof BreakStatement)
        contextOfInterest = MCUDAUtils.getBreakContext((BreakStatement)nextstmt);
      else if (nextstmt instanceof Case)
        contextOfInterest = MCUDAUtils.getCaseContext((Case)nextstmt);

      List<Statement> valList = contextToStmtsMap.get(contextOfInterest);
      if(valList == null)
        contextToStmtsMap.put(contextOfInterest, 
            new ChainedList<Statement>().addLink(nextstmt));
      else
        valList.add(nextstmt);
    }


    // First thing we need to deal with is returns.  We want a procedure
    // with a single return as the last statement of the procedure, 
    // with no control dependence on anything but function entry, but 
    // without introducing unstructured control flow if possible.

    if(contextToStmtsMap.containsKey(proc.getBody()))
      doProcedure(proc, contextToStmtsMap.get(proc.getBody()));
    /*
       for( CompoundStatement context : contextToStmtsMap.keySet())
       {
       Traversable control = context.getParent();
       List<Statement> Irregs = contextToStmtsMap.get(control);
       if(control instanceof Procedure)
    //Already taken care of
    continue;
    else if (control instanceof SwitchStatement)
    //deal with switches, cases and breaks in list
    continue;
    //doSwitch(control, contextToStmtsMap.get(control));
    else if (control instanceof Loop)
    //deal with loops, continues and breaks in list
    continue;
    //doLoop(control, contextToStmtsMap.get(control));
    else
    //Should be it
    throw new InternalError("Don't understand irregular flow context");
       }
       */
  }
}



