package fcuda.common;
import fcuda.*;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;


public class AnnotData
{
  private LinkedList<Statement> mStmtList;

  //Compound stmt representing 
  //if(enable)
  //	mTaskBody
  private CompoundStatement mFunctionBody;

  //See above comment - actual useful part of fcuda core (guarded by enable signal in mFunctionBody)
  private CompoundStatement mTaskBody;

  private FunctionCall mFcudaCore;

  HashSet<IDExpression> mTaskArgSet;
  LinkedList<IDExpression> mTaskArgs; 
  LinkedList<Declaration> mTaskDecls;

  public void clear()
  {	
    mStmtList.clear();

    mTaskArgSet.clear();
    mTaskDecls.clear(); 
    mTaskArgs.clear(); 
  }

  public AnnotData()
  {
    mStmtList = new LinkedList<Statement>();

    mTaskArgSet = new HashSet<IDExpression>();
    mTaskDecls = new LinkedList<Declaration>();
    mTaskArgs = new LinkedList<IDExpression>();

    clear();	
  }

  public void addStatement(Statement stmt)
  {
    mStmtList.add(stmt);	
  }

  public List<Statement> getStatementList()
  {
    return mStmtList;
  }

  public void setBody(CompoundStatement c)
  {
    mFunctionBody = c;
  }
  public CompoundStatement getBody()
  {
    return mFunctionBody;
  }

  public void setTaskBody(CompoundStatement c)
  {
    mTaskBody = c;
  }

  public CompoundStatement getTaskBody()
  {
    return mTaskBody;
  }

  public void addArgsInfo(Set<IDExpression> taskArgSet, 
      List<IDExpression> taskArgs, List<Declaration> taskDecls)
  {
    mTaskArgSet.addAll(taskArgSet);
    mTaskArgs.addAll(taskArgs);
    mTaskDecls.addAll(taskDecls);
  }

  public void getArgsInfo(Set<IDExpression> taskArgSet, 
      List<IDExpression> taskArgs, List<Declaration> taskDecls)
  {
    taskArgSet.addAll(mTaskArgSet);
    taskArgs.addAll(mTaskArgs);
    taskDecls.addAll(mTaskDecls);
  }

  public void setFcudaCore(FunctionCall c)
  {
    mFcudaCore = c;
  }

  public FunctionCall getFcudaCore()
  {
    return mFcudaCore;
  }
}
