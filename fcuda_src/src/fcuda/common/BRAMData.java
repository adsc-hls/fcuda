package fcuda.common;
import fcuda.*;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;


public class BRAMData
{
  private HashSet<FunctionCall> mFCUDACores;

  private int dims;

  public BRAMData()
  {
    mFCUDACores = new HashSet();
    clear();
  }

  public void clear()
  {
    mFCUDACores.clear();
  }

  public void addFCUDACore(FunctionCall coreCall)
  {
    mFCUDACores.add(coreCall);
  }

  public HashSet<FunctionCall> getFCUDACores()
  {
    return mFCUDACores;
  }

  public void setDim(int dim) { 
    dims = dim;
  }
}
