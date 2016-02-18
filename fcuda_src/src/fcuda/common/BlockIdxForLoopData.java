package fcuda.common;
import fcuda.*;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;


public class BlockIdxForLoopData
{
  private ExpressionStatement mExprStmt;

  public BlockIdxForLoopData()
  {
    clear();
  }

  public void clear()
  {
  }

  public void setAssignmentStmt(ExpressionStatement stmt)
  {
    mExprStmt = stmt;
  }	

  public ExpressionStatement getAssignmentStmt()
  {
    return mExprStmt;
  }	
}
