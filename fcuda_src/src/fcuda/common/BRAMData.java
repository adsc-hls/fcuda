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
