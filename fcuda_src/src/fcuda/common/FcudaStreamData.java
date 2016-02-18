package fcuda.common;
import fcuda.*;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;


public class FcudaStreamData
{
  private LinkedList<ForLoop> loops;
  private LinkedList<IDExpression> loopVars;
  private LinkedList<IDExpression> loopBounds;
  private LinkedList<IDExpression> loopUpdates;
  private IDExpression kernelName;
  private HashSet<IDExpression> mConstMemIDs;

  public FcudaStreamData(Procedure kernel, ForLoop streamLoop, IDExpression streamLoopVar,
      IDExpression streamLoopBound, IDExpression streamLoopUpdate)
  {
    loops = new LinkedList<ForLoop>();
    loops.add(streamLoop);
    //loopVar = streamLoopVar;
    loopVars = new LinkedList<IDExpression>();
    loopVars.add(streamLoopVar);
    loopBounds = new LinkedList<IDExpression>();
    loopBounds.add(streamLoopBound);
    //loopUpdate = streamLoopUpdate;
    loopUpdates = new LinkedList<IDExpression>();
    loopUpdates.add(streamLoopUpdate);
    kernelName = kernel.getName();
    mConstMemIDs = new HashSet<IDExpression>();
    mConstMemIDs.clear();
  }

  public void addStream(ForLoop streamLoop, IDExpression streamLoopVar,
      IDExpression streamLoopBound, IDExpression streamLoopUpdate)
  {
    loops.add(streamLoop);
    loopVars.add(streamLoopVar);
    loopBounds.add(streamLoopBound);
    loopUpdates.add(streamLoopUpdate);
  }


  public void addConstMemID(IDExpression memID)
  {
    mConstMemIDs.add(memID);
  }

  public HashSet<IDExpression> getConstMemIDs()
  {
    return mConstMemIDs;
  }

  public LinkedList<IDExpression> getLoopVar()
  {
    return loopVars;
  }

  public LinkedList<IDExpression> getLoopBound()
  {
    return loopBounds;
  }

  public LinkedList<IDExpression> getLoopUpdate()
  {
    return loopUpdates;
  }

  public IDExpression getKernelName()
  {
    return kernelName;
  }

  public LinkedList<ForLoop> getForLoop()
  {
    return loops;
  }
}

