package fcuda.common;
import fcuda.*;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;


public class WhileLoopData
{
  private Expression mCondition;
  private IfStatement mGuardIf;
  private IDExpression mGuardVar;
  private Statement mGuardInitStmt;
  private Statement mInitLoopVar;

  public void setWhileLoopCondition(Expression t)
  {
    mCondition = t;
  }

  public Expression getWhileLoopCondition()
  {
    return mCondition;
  }

  public void setGuardIf(IfStatement s)
  {
    mGuardIf = s;
  }

  public void setGuardVar(IDExpression v)
  {
    mGuardVar = v;
  }

  public IDExpression getGuardVar()
  {
    return mGuardVar;
  }

  public IfStatement getGuardIf()
  {
    return mGuardIf;
  }

  public void setGuardInitStmt(Statement t)
  {
    mGuardInitStmt = t;
  }

  public Statement getGuardInitStmt()
  {
    return mGuardInitStmt;
  }

  public void setInitLoopVar(Statement t)
  {
    mInitLoopVar = t;
  }

  public Statement getInitLoopVar()
  {
    return mInitLoopVar;
  }
}
