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

import fcuda.utils.*;
import fcuda.common.*;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;

/**
 * Duplicate every procedure
 */

public class DuplicateForFCUDAByCloning extends KernelTransformPass
{
  private Procedure mProcedure;
  private int numParallelThreadBlocks;
  private int numCores;
  // Add original procedures to this Set when they are renamed
  private Set<Procedure> renamedProcs;

  private HashMap<Procedure, Procedure> removedProcs;
  private LinkedList<Procedure> constMemKern;
  // Add *new* FunctionCalls to this Set when they are created
  private Set<FunctionCall> renamedCalls;

  // Map orig => new
  private Map<Procedure,Procedure> procMap;

  private boolean forLoopHandled;

  // private Set<FunctionCall> doneFunctions();
  public DuplicateForFCUDAByCloning(Program program)
  {
    super(program);
    System.out.println("Starting....");
    renamedProcs = new HashSet<Procedure>();
    removedProcs = new HashMap<Procedure, Procedure>();
    constMemKern = new LinkedList<Procedure>();
    renamedCalls = new HashSet<FunctionCall>();
    procMap = new HashMap<Procedure, Procedure>();
    clear();
  }

  public void clear()
  {
    renamedProcs.clear();
    renamedCalls.clear();
    procMap.clear();
  }

  public String getPassName()
  {
    return new String("[DuplicateForFCUDAByCloning-FCUDA]");
  }

  private void addProcedureToTaskMap(Procedure toplevel, Procedure nproc) {
    System.out.println("Test add to task map: " + toplevel + " => " + nproc);
    FCUDAutils.addTaskMapping(toplevel.getSymbolName(), nproc);
    List<FcudaAnnotation> procAnnots = nproc.getAnnotations(FcudaAnnotation.class);
    String annotType    = (String) procAnnots.get(0).get("fcuda");
    FCUDAutils.setTaskType(nproc, annotType);
  }

  private boolean procedureInTaskMap(Procedure p) {
    HashMap<String,List<Procedure>> tm = FCUDAutils.getTaskMap();

    Collection<List<Procedure>> col = tm.values();

    for (List<Procedure> proList : col) {
      for (Procedure pro : proList) {
        if (p == pro) {
          return true;
        }
      }
    }
    return false;
  }

  public boolean checkStmtInList(Statement stmt, LinkedList<Statement> l)
  {
    for (Statement s : l) {
      if (stmt.toString().equals(s.toString()))
        return true;
    }
    return false;
  }

  public boolean handleProc(Procedure proc, int kernelNum, int coreNum, int numBlocks,
      Procedure orig_proc) {

    CompoundStatement body = proc.getBody();
    DepthFirstIterator<Traversable> dfi = new DepthFirstIterator(body);

    LinkedList<Statement> listInitStmt = FCUDAGlobalData.getInitializationBlkXStmt(orig_proc);

    // iterate over every statement in the body
    while (dfi.hasNext()) {
      Traversable t = dfi.next();

      // Handle function calls
      if (t instanceof FunctionCall) {

        // If this function call already has a replacement, rename it

        // If no replacement yet, 
        //   Create a replacement 
        //   Setup the replacement
        //   Add to 'replaced' list
        //   Recurse to that function: handleProc(function.getProcedure())
        //   Note that recursion is not actually used -it's a while loop

        FunctionCall f = (FunctionCall)t;

        // Skip this call if already renamed
        if (renamedCalls.contains(f)) {
          continue;
        }
        // If we have already cloned this procedure
        if (renamedProcs.contains(f.getProcedure())) {
          Traversable ch = f.getChildren().get(0);
          System.out.println("Got the child[0]: " + ch);

          // Get the cloned procedure
          Procedure othProc = procMap.get(f.getProcedure());

          // remap to point to other proc
          f.setChild(0,new Identifier(othProc)); 
          renamedCalls.add(f);
          // Reset
          return false;

        } else {
          // We need to clone the procedure
          Procedure oldProc = f.getProcedure();
          // Pipelining stmts should not be duplicated anymore
          if (removedProcs.containsKey(oldProc)) {
            Procedure newProc = removedProcs.get(oldProc);
            if (renamedProcs.contains(newProc)) {
              f.setChild(0, new Identifier(newProc)); 
              renamedCalls.add(f);
              procMap.put(f.getProcedure(), newProc);
              continue;
            }
          }
          Procedure newProc = f.getProcedure().clone();

          // Look up the procedure in the task-map
          // If it is there, then we need to add this new one to the mapping
          // Along with the new top-level function name...
          //

          // we need to update the task mapping so that we can 
          // correctly remove the __syncthreads calls later
          if (procedureInTaskMap(f.getProcedure())) {
            addProcedureToTaskMap(proc, newProc);
          }
          // Give it a new name
          String newName = getStringName(kernelNum, coreNum, f.getProcedure());
          NameID newNameID = SymbolTools.getNewName(newName, f.getProcedure().getParent());
          newProc.setName(newNameID);

          Expression ch = (Expression)f.getChildren().get(0);
          System.out.println("Got the child[1]: " + ch);

          // Add the cloned procedure to source code
          if (!constMemKern.contains(oldProc) && 
              (f.getProcedure().getParent() instanceof TranslationUnit)) {
            System.out.println("Added declaration of cloned procedure");
            TranslationUnit tr = (TranslationUnit)f.getProcedure().getParent();
            tr.addDeclarationBefore(f.getProcedure(), newProc); 
          } else {
            System.out.println("Error!!!");
          }

          // Update f to point to cloned function

          // Need to create an expression from this
          f.setChild(0, new Identifier(newProc)); 
          // update map, list of completed functions
          renamedProcs.add(f.getProcedure());
          renamedCalls.add(f);
          procMap.put(f.getProcedure(), newProc);
          removedProcs.put(oldProc, newProc);
          System.out.println("Done...................");
          return false;
        }
      } else if (t instanceof ForLoop) {
        ForLoop f =  (ForLoop)t;
        if (forLoopHandled == false) {
          forLoopHandled = handleForLoop(f, coreNum, orig_proc, proc);
        }
      } else if (t instanceof ExpressionStatement)  {
        ExpressionStatement es = (ExpressionStatement)t;
        Annotation toRemove = null;
        if (es.getAnnotations() != null) {
          for (Annotation an : es.getAnnotations()) {
            if (an.get("idx") != null && Integer.parseInt((String)an.get("idx"))  == 0) {
              AssignmentExpression as = ((AssignmentExpression)es.getExpression());
              Expression e = as.getRHS();
              as.setRHS(new BinaryExpression( e.clone(), BinaryOperator.ADD , new IntegerLiteral(coreNum))); 
              toRemove = an;
              break;
            }
          }

          if (toRemove != null){
            es.getAnnotations().remove(toRemove);
          }
        }                               
        if (checkStmtInList(es, listInitStmt)) {
          ExpressionStatement newStmt = new ExpressionStatement(
              new AssignmentExpression(
                ((BinaryExpression)((ExpressionStatement)es).getExpression()).getLHS().clone(),
                AssignmentOperator.NORMAL,
                new BinaryExpression(
                  ((BinaryExpression)((ExpressionStatement)es).getExpression()).getRHS().clone(),
                  BinaryOperator.ADD,
                  new BinaryExpression(
                    new IntegerLiteral(coreNum),
                    BinaryOperator.MULTIPLY,
                    new IntegerLiteral(numParallelThreadBlocks)))));
          es.swapWith(newStmt);
        }
      } 
    }
    return true;
  }

  // *TAN* handle BlockIdxForLoop
  private boolean handleForLoop(ForLoop forLoop, int coreNum, Procedure orig_proc, Procedure proc)
  {	
    ForLoop orig_forLoop = FCUDAGlobalData.getInnermostBlockIdxForLoop(orig_proc);
    if (orig_forLoop == null)
      return false;
    if(orig_forLoop.getInitialStatement().toString().equals(forLoop.getInitialStatement().toString()))
    {
      Expression stepExpr = forLoop.getStep();
      if(stepExpr instanceof BinaryExpression)
        ((BinaryExpression)stepExpr).setRHS(new IntegerLiteral(numParallelThreadBlocks));
      else
        if(stepExpr instanceof UnaryExpression)
        {
          BinaryExpression newStepExpr = new AssignmentExpression(
              (Expression)((UnaryExpression)stepExpr).getExpression().clone(),
              AssignmentOperator.ADD,
              new IntegerLiteral(numParallelThreadBlocks)
              );
          forLoop.setStep(newStepExpr);
        }
        else
          Tools.exit("What kind of expression is step expr "+stepExpr.toString()+" in for loop : \n"
              + forLoop.toString());

      ExpressionStatement orig_assignStmt = FCUDAGlobalData.getBlockIdxForLoopAssignmentStmt(orig_forLoop);
      ExpressionStatement assignStmt = null;
      CompoundStatement bodyStmt = (CompoundStatement)forLoop.getBody();
      DepthFirstIterator<Traversable> dfi = new DepthFirstIterator(bodyStmt);

      // iterate over every statement in the body
      while (dfi.hasNext()) {
        Traversable t = dfi.next();
        if (t instanceof ExpressionStatement)
        {
          assignStmt = (ExpressionStatement)t;
          if (assignStmt.toString().equals(orig_assignStmt.toString()))
          {
            break;
          }
        }
      }

      ExpressionStatement newStmt = new ExpressionStatement(new AssignmentExpression(
            MCUDAUtils.getBidxID(0),
            AssignmentOperator.NORMAL,
            new BinaryExpression(
              (Expression)WrapBlockIdxLoop.getBidxCounterID(0, (SymbolTable)mProcedure.getBody()).clone(),
              BinaryOperator.ADD,
              new IntegerLiteral(coreNum)
              )
            )
          );

      bodyStmt.addStatementBefore(assignStmt, newStmt);
      assignStmt.detach();
      return true;
    }
    return false;
  }

  public String getStringName(int kernelNum, int coreNum, Procedure p) {
    return p.getName().getName() +  "_core" + Integer.toString(coreNum);
  }

  public void runPass(Procedure proc, int numBlocks)
  {
    mProcedure = proc;
    int kernelNum = 0;
    int coreNum = 0;

    for (coreNum = 0; coreNum < numCores; coreNum++) {
      // clone the original proedure
      Procedure newProc = proc.clone();


      // Set up new name
      String newName = getStringName(kernelNum, coreNum, proc);
      NameID newNameID = SymbolTools.getNewName(newName, proc.getParent());

      CommentAnnotation ca = new CommentAnnotation("FCUDA_CORE_ID: " + newNameID.toString());
      ca.setOneLiner(true);

      // rename the new procedure
      newProc.setName(newNameID);

      newProc.annotateBefore(ca);

      // add newProc to the translationUnit
      if ((proc.getParent() instanceof TranslationUnit)) {
        TranslationUnit tr = (TranslationUnit)proc.getParent();
        tr.addDeclarationBefore(proc, newProc); 
      } else {
        System.out.println("Error!!!");
      }

      // mark proc as completed
      renamedProcs.add(proc);
      procMap.put(proc, newProc);

      // add to ProceduresNeedingPragmas 
      // *TAN* disable for now
      //FCUDAGlobalData.addProcedureNeedingPragma(newProc);

      // Finally, recursively rename all function calls and add them to respective translationUnits
      forLoopHandled = false;
      while (!handleProc(newProc, kernelNum, coreNum, numBlocks, proc));
      // before we start on next core, clear out data structures...
      clear();
    }
    for (Procedure p : removedProcs.keySet())
      p.detach();
    removedProcs.clear();

    if (FCUDAGlobalData.getConstMemKern() == null ||
        !proc.getName().equals(FCUDAGlobalData.getConstMemKern())) {
      proc.detach();
      for (Procedure p : constMemKern)
        p.detach();
    } else
      constMemKern.add(proc);

  } // runPass()

  public void transformProcedure(Procedure proc)
  {
    numCores = 1;
    for (Annotation annot : proc.getAnnotations()) {
      if (annot.get("fcuda") == "coreinfo") {
        String numCoresStr = annot.get("num_cores");
        if (numCoresStr != null)
          numCores = Integer.parseInt(numCoresStr);
      }
    }

    //  Get the number of threadBlocks
    numParallelThreadBlocks = FCUDAGlobalData.getNumParallelThreadBlocks(proc);

    System.out.println("Beginning: there are " + Integer.toString(numParallelThreadBlocks)  + " threadBlocks");

    // Duplicate the cores
    runPass(proc, numParallelThreadBlocks);

  }

}


