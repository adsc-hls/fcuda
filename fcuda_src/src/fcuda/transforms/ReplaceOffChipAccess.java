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

package fcuda.transforms;
import fcuda.*;

import java.util.*;

import cetus.hir.*;
//import cetus.analysis.*;
//import cetus.exec.*;
//import cetus.transforms.*;

public class ReplaceOffChipAccess
{
  // *AP*
  // COMMENTED OUT (NOT used currently)
  // *AP*
  /*

     private PrepareForSplit mSplitPrepare;
     private Procedure mProcedure;
     HashSet<Statement> mIgnoreStmts;


     private HashMap<IDExpression, IDExpression> mPtr2Bram;

     private HashMap<ArrayAccess, ArrayAccess> mReplaceAccesses;

     ReplaceOffChipAccess()
     {
     mIgnoreStmts = new HashSet<Statement>();
     mPtr2Bram = new HashMap<IDExpression, IDExpression>();
     mReplaceAccesses = new HashMap<ArrayAccess, ArrayAccess>();
     clear();
     }

     public void clear()
     {
     mPtr2Bram.clear();	
     mIgnoreStmts.clear();
     mReplaceAccesses.clear();
     }

     public void initialize(Procedure proc, PrepareForSplit inp)
     {
     mSplitPrepare = inp;
     mProcedure = proc;	
     }

     public void addPtr2BramEntry(IDExpression ptr, IDExpression bram)
     {
     mPtr2Bram.put(ptr, bram);
     }

     public IDExpression getBramEntry(IDExpression ptrId)
     {
     if(!mPtr2Bram.containsKey(ptrId))
     Tools.exit("No BRAM found for pointer "+ptrId.toString());
     return mPtr2Bram.get(ptrId);
     }

     private void removeIgnoredStmts()
     {
  //Iterator iter = mIgnoreStmts.iterator();
  for(Statement tmp : mIgnoreStmts)
  {
  //Statement tmp = (Statement)iter.next();
  tmp.detach();
  }
  mIgnoreStmts.clear();
     }

     public void doActualReplace()
     {
     removeIgnoredStmts();
     Set<Map.Entry<ArrayAccess, ArrayAccess>> defSet = mReplaceAccesses.entrySet();
  //Iterator printIter = defSet.iterator();
  for(Map.Entry<ArrayAccess, ArrayAccess> mapEntry : defSet)
  {
  Tools.replaceAll(mapEntry.getKey(), mapEntry.getKey(), mapEntry.getValue());
  }
     }

  //Last argument is true if the pointer passed is __shared__ variable in CUDA
  public void replaceOffChipAccess(String ptrName, int dir, boolean isSharedMem)
  {
  if(!isSharedMem)
  findSharedMemAccess(ptrName, dir);

  DepthFirstIterator iter = new DepthFirstIterator(mProcedure);
  iter.pruneOn(ArrayAccess.class);

  while(iter.hasNext()) 
  {
    ArrayAccess currAccess = null;
    try
    {
      currAccess = iter.next(ArrayAccess.class);
    }
    catch(NoSuchElementException e)
    {
      break;
    }
    if(mIgnoreStmts.contains(FCUDAutils.getClosestParentStmt(currAccess)))
      continue;

    if(!(currAccess.getArrayName() instanceof IDExpression))
      Tools.exit("What kind of expression is "+currAccess.getArrayName().toString());
    IDExpression currId = (IDExpression)currAccess.getArrayName();
    if(currId.getSymbol().getSymbolName().equals(ptrName))
    {
      if(isSharedMem)
      {
        List<Expression> indices = currAccess.getIndices();
        if(currAccess.getNumIndices() < 2)
          indices.add(0, MCUDAUtils.Tidx.getId(1));
        ArrayAccess newArrayAccess = new ArrayAccess(getBramEntry(currId), indices);
        mReplaceAccesses.put(currAccess, newArrayAccess);
        //Tools.replaceAll((Traversable)currAccess.getParent(), currAccess, newArrayAccess);
        //System.out.println("Shared  array "+ptrName+" access "+currAccess.toString() + " replaced with "+newArrayAccess);
      }
      else
      {
        Expression c2 = mSplitPrepare.getCoeffC2(currAccess);
        Expression c4 = mSplitPrepare.getCoeffC4(currAccess);

        Expression tyExpr = null, txExpr = null;
        if(c2 != null)
          tyExpr = new BinaryExpression((Expression)MCUDAUtils.getTidID(1)
              , BinaryOperator.ADD
              , (Expression)c2.clone());
        else
          tyExpr = (Expression)MCUDAUtils.getTidID(1);

        if(c4 != null)
          txExpr = new BinaryExpression((Expression)MCUDAUtils.getTidID(0)
              , BinaryOperator.ADD
              , (Expression)c4.clone());
        else
          txExpr = (Expression)MCUDAUtils.getTidID(0);

        LinkedList<Expression> indices = new LinkedList();
        indices.add(tyExpr);
        indices.add(txExpr);
        ArrayAccess newArrayAccess = new ArrayAccess(getBramEntry(currId), indices);
        mReplaceAccesses.put(currAccess, newArrayAccess);
      }
    }
  }
  }

public void findSharedMemAccess(String ptrName, int dir)
{
  DepthFirstIterator iter = new DepthFirstIterator(mProcedure);
  iter.pruneOn(ArrayAccess.class);

  while(iter.hasNext()) 
  {
    ArrayAccess currAccess = null;
    try
    {
      currAccess = iter.next(ArrayAccess.class);
    }
    catch(NoSuchElementException e)
    {
      break;
    }

    //System.out.println("Testing shared for "+currAccess.toString());
    //Traversable tmpAccess = currAccess.getParent();
    //while(tmpAccess != null)
    //{
    //System.out.println("Parent "+currAccess.getParent());
    //tmpAccess = tmpAccess.getParent();
    //}
    if(FCUDAutils.isSharedMem(currAccess))
    {
      if(!(currAccess.getArrayName() instanceof IDExpression))
        Tools.exit("What kind of expression is : "+currAccess.getArrayName());
      VariableDeclaration arrayDecl = FCUDAutils.getVariableDeclaration((IDExpression)(currAccess.getArrayName()));
      Statement declStmt = FCUDAutils.getClosestParentStmt(arrayDecl);
      mIgnoreStmts.add(declStmt); // *AP* This is not correct for shared arrays that are only used within compute

      //Check if parent statement accesses the pointer
      Statement tmpStmt = FCUDAutils.getClosestParentStmt(currAccess);

      DepthFirstIterator iter2 = new DepthFirstIterator(tmpStmt);
      iter2.pruneOn(ArrayAccess.class);
      while(iter2.hasNext()) 
      {
        ArrayAccess ptrAccess = null;
        try
        {
          ptrAccess = iter2.next(ArrayAccess.class);
        }
        catch(NoSuchElementException e)
        {
          break;
        }
        if(!(ptrAccess.getArrayName() instanceof IDExpression))
          Tools.exit("What kind of expression is "+ptrAccess.getArrayName().toString());
        IDExpression ptrId = (IDExpression)ptrAccess.getArrayName();
        if(ptrId.getSymbol().getSymbolName().equals(ptrName))
        {
          if(!(tmpStmt instanceof ExpressionStatement) || 
              !(((ExpressionStatement)tmpStmt).getExpression() instanceof AssignmentExpression)
            )
            Tools.exit("What kind of statement is this ? "+tmpStmt.toString());
          AssignmentExpression asgExpr = (AssignmentExpression)(((ExpressionStatement)tmpStmt).getExpression());	

          //FIXME: extremely limited - only handles statements of form
          // Ashared[threadIdx.y][threadIdx.x] = ptr[Address];
          boolean readFlag = (asgExpr.getLHS().equals(currAccess) && asgExpr.getRHS().equals(ptrAccess));
          boolean writeFlag = (asgExpr.getLHS().equals(ptrAccess) && asgExpr.getRHS().equals(currAccess));
          if(!readFlag && !writeFlag)
            Tools.exit("Cannot handle expression for BRAM transfer "+tmpStmt.toString());
          //FIXME: check if currAccess is of form Ashared[threadIdx.y][threadIdx.x]

          if(writeFlag && dir == 1)
          {
            addPtr2BramEntry(((IDExpression)currAccess.getArrayName()), 
                getBramEntry((IDExpression)(ptrAccess.getArrayName()))
                );
            mIgnoreStmts.add(tmpStmt);
            replaceOffChipAccess(((IDExpression)currAccess.getArrayName()).getSymbol().getSymbolName(),
                dir,
                true);
          }

          if(readFlag && dir == 0)
          {
            System.out.println("Found required stmts "+tmpStmt.toString());
            addPtr2BramEntry(((IDExpression)currAccess.getArrayName()), 
                getBramEntry((IDExpression)(ptrAccess.getArrayName()))
                );
            mIgnoreStmts.add(tmpStmt);
            replaceOffChipAccess(((IDExpression)currAccess.getArrayName()).getSymbol().getSymbolName(),
                dir,
                true);
          }
        }
      }
    }
  }
}

*/
}

