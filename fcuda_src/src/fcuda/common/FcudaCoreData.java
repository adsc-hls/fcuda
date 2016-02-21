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


public class FcudaCoreData
{
  public static int NO_TYPE = 0x0; 
  public static int COMPUTE_TYPE = 0x1; 
  public static int TRANSFER_TYPE = 0x2; 

  private IDExpression mEnableId;
  private int mNumCores;
  private int mType;
  private String mName;
  private LinkedList<Integer> mCommonArgsIndex;

  public FcudaCoreData()
  {
    mType = NO_TYPE;
    mCommonArgsIndex = new LinkedList<Integer>();
    clear();
    mNumCores = 1;
  }

  public void clear()
  {
    mCommonArgsIndex.clear();
  }

  public void setEnableSignal(IDExpression expr)
  {
    mEnableId = expr;
  }

  public IDExpression getEnableSignal()
  {
    return mEnableId;
  }

  public void setNumCores(int v)
  {
    mNumCores = v;
  }

  public int getNumCores()
  {
    return mNumCores;
  }

  public void setType(int t)
  {
    mType = t;
  }

  public int getType()
  {
    return mType;
  }

  public void setName(String s)
  {
    mName = s;
  }

  public String getName()
  {
    return mName;
  }

  public void addCommonArgsIndex(LinkedList<Integer> argsIndex)
  {
    mCommonArgsIndex.addAll(argsIndex);
  }

  public LinkedList<Integer> getCommonArgsIndex()
  {
    return mCommonArgsIndex;
  }

  // *AP* Update indexes after <position> by <update>
  public void updateCommonArgsIndex(int position, int update)
  {
    LinkedList<Integer> newIndex = new LinkedList<Integer>();
    for (Integer idx : mCommonArgsIndex) {
      if(idx.intValue() > position)
        newIndex.add(idx.intValue()+update);
      else 
        newIndex.add(idx);
    }
    mCommonArgsIndex.clear();
    mCommonArgsIndex.addAll(newIndex);
  }


  public void createCopy(FcudaCoreData t)
  {
    mCommonArgsIndex = new LinkedList();
    mCommonArgsIndex.clear();
    mCommonArgsIndex.addAll(t.getCommonArgsIndex());

    mName = t.getName();
    mEnableId = t.getEnableSignal();
    mNumCores = t.getNumCores();
    mType = t.getType();
  }
}

