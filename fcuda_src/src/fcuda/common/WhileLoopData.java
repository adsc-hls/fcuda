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
