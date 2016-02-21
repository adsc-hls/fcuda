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
import fcuda.common.*;
import java.util.*;
import fcuda.utils.*;
import fcuda.ir.*;
import fcuda.analysis.*;
import cetus.hir.*;
import cetus.exec.*;

/**
 * Identify scalars which are defined in one ThreadLoop
 * and used in another ThreadLoop within a task,
 * then convert it to array form so that each thread
 * has its own copies of the scalars.
 */

public class PrivatizeScalarsInThreadLoops extends KernelTransformPass
{
  private Procedure mProcedure;
  Set<Symbol> scalarsToPrivatize = new HashSet<Symbol>();
  public String getPassName()
  {
    return new String("[PrivatizeScalarsInThreadLoops - FCUDA]");
  }

  public PrivatizeScalarsInThreadLoops(Program program)
  {
    super(program);
  }

  public void privatizeScalars(Procedure proc)
  {
    DataDepAnalysis datadep = new DataDepAnalysis(proc);
    HashMap<CompoundStatement, List<Set<Symbol>>> defUseCollection = new HashMap<CompoundStatement,
      List<Set<Symbol>>>();
    for (CompoundStatement cs: MCUDAUtils.getThreadLoops(proc.getBody())) {
      System.out.println("THREADLOOP: " + cs);
      Set<Symbol> defs = Tools.getDefSymbol(cs);
      Set<Symbol> uses = Tools.getUseSymbol(cs);
      defUseCollection.put(cs, new ChainedList<Set<Symbol>>().addLink(defs).addLink(uses));
    }

    for (CompoundStatement cs: MCUDAUtils.getThreadLoops(proc.getBody())) {
      Set<Symbol> defSet = defUseCollection.get(cs).get(0);
      Set<Symbol> elementsToRemove = new HashSet<Symbol>();
      for (Symbol def: defSet) {
        // *TAN* we need to add more checks here to remove
        // as much unnecessary vars before proceeding
        if (def instanceof AccessSymbol) {
          Symbol defBase = ((AccessSymbol)def).getBaseSymbol();
          // We should not process global varibles.
          // Only local variables are the concern here.
          if (SymbolTools.isGlobal((VariableDeclarator)defBase))
            elementsToRemove.add(def);
          continue;
        }

        IDExpression defID = ((VariableDeclarator)def).getID();
        // Exclude CUDA specific variables (threadIdx., blockIdx., ...)
        // Also exclude non-scalar variables (array form ...)
        if (!datadep.isThrDepVar(defID) || !FCUDAutils.isScalar((VariableDeclarator)def)) {
          elementsToRemove.add(def);
          continue;
        }

        VariableDeclaration defDecl = Tools.getAncestorOfType(defID, VariableDeclaration.class);
        HashSet<Specifier> defSpecifiers = new HashSet<Specifier>();
        defSpecifiers.addAll(defDecl.getSpecifiers());
        if (defSpecifiers.contains(Specifier.SHARED)) {
          elementsToRemove.add(def);
          continue;
        }
      }
      defSet.removeAll(elementsToRemove);
      if (defSet.isEmpty())
        continue;
      System.out.println("DEFSET: " + defSet);
      for (CompoundStatement otherThreadLoop: MCUDAUtils.getThreadLoops(proc.getBody())) {
        if (!cs.equals(otherThreadLoop)) {
          System.out.println("Find UseSet of: " + otherThreadLoop);
          Set<Symbol> useSet = defUseCollection.get(otherThreadLoop).get(1);
          Set<Symbol> defSetWithin = defUseCollection.get(otherThreadLoop).get(0);
          System.out.println("USESET: " + useSet);
          System.out.println("DEFSET WITHIN LOOP: " + defSetWithin);

          useSet.removeAll(defSetWithin);
          Set<Symbol> remainSet = new HashSet<Symbol>();
          remainSet.addAll(useSet);
          remainSet.retainAll(defSet);
          System.out.println("REMAINSET: " + remainSet);
          /* if a scalar is defined in one ThreadLoop
           * and is used in another ThreadLoop, we will
           * privatize it
           */
          if (!remainSet.isEmpty())
            scalarsToPrivatize.addAll(remainSet);
        }
      }
    }
  }

  public void convertToArray()
  {
    System.out.println("Scalars to be privatized: " + scalarsToPrivatize);
    int numDims = FCUDAGlobalData.getKernTblkDim(mProcedure);
    IDExpression bdimXid, bdimYid, bdimZid;
    VariableDeclarator bdimXtor = null, bdimYtor = null, bdimZtor = null;

    bdimXid = (IDExpression) new NameID("BLOCKDIM_X" + "_" + mProcedure.getSymbolName());
    bdimXtor = new VariableDeclarator(bdimXid);
    if(numDims > 1) {
      bdimYid = (IDExpression) new NameID("BLOCKDIM_Y" + "_" + mProcedure.getSymbolName());
      bdimYtor = new VariableDeclarator(bdimYid);
    }
    if(numDims > 2) {
      bdimZid = (IDExpression) new NameID("BLOCKDIM_Z" + "_" + mProcedure.getSymbolName());
      bdimZtor = new VariableDeclarator(bdimZid);
    }

    // Create array specifier and convert to arrays
    LinkedList<Expression> arrayDims = new LinkedList<Expression>();
    if (numDims == 3)
      arrayDims.add(new Identifier(bdimZtor));
    if (numDims > 1)
      arrayDims.add(new Identifier(bdimYtor));
    arrayDims.add(new Identifier(bdimXtor));
    for(Symbol scalar : scalarsToPrivatize) {
      VariableDeclarator currDeclor = (VariableDeclarator)scalar;
      VariableDeclaration currDecl = FCUDAutils.getVariableDeclaration(currDeclor.getID());
      VariableDeclaration blockDecl = FCUDAutils.createSharedArrayDeclaration(currDecl, currDeclor, arrayDims);
      ((CompoundStatement)IRTools.getAncestorOfType(currDeclor, CompoundStatement.class)).addDeclarationAfter(currDecl,  blockDecl);

      // Remove scalar declaration
      Statement declStmt = (Statement)IRTools.getAncestorOfType(currDecl, Statement.class);
      declStmt.detach();

      // Replace scalar identifier with array identifier
      Expression blockAccess = FCUDAutils.getThreadIdxArrayAccess(blockDecl, numDims);
      Traversable t = (Traversable)program;
      IRTools.replaceAll(t, (Expression)currDeclor.getID(), blockAccess);
    }
    arrayDims.clear();
  }

  public void transformProcedure(Procedure proc)
  {
    // Skip this if the current proc is stream proc
    Expression constMemKern = FCUDAGlobalData.getConstMemKern();
    if (constMemKern != null) {
      if (proc.getName().toString().equals(constMemKern.toString()))
        return;
    }
    mProcedure = proc;
    scalarsToPrivatize.clear(); //clear old info before moving to next kernel
    List<Procedure> tskList = FCUDAutils.getTaskMapping(proc.getSymbolName());
    if (tskList != null) {
      for (Procedure task: tskList) {
        if (FCUDAutils.getTaskType(task).equals("compute"))
          privatizeScalars(task);
      }
    }
    convertToArray();
  }
}
