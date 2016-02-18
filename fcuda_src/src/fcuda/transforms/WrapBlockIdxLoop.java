package fcuda.transforms;

import java.util.*;

import fcuda.common.*;
import fcuda.ir.*;
import fcuda.utils.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;

/**
 *
 * Pass to create blockIdx loop over CUDA statements
 *
 */

public class WrapBlockIdxLoop extends KernelTransformPass
{
  private Procedure mProcedure;

  public WrapBlockIdxLoop(Program program)
  {
    super(program);
    clear();
  }

  public void clear()
  {

  }

  public String getPassName()
  {
    return new String("[WrapBlockIdxLoop-FCUDA]");
  }

  public static IDExpression getBidxCounterID()
  {
    return new NameID("blockIdx_loop");
  }

  public static IDExpression getNumCoresID()
  {
    return new NameID("num_cores");
  }

  public static IDExpression getCoreID()
  {
    return new NameID("core_id");
  }

  public static VariableDeclaration getBidxCounterDecl()
  {
    VariableDeclaration decl = new VariableDeclaration(
        new UserSpecifier(new NameID("dim3")),
        new VariableDeclarator(getBidxCounterID()));
    return decl;
  }

  public static Expression getBidxCounterID(int i, SymbolTable body)
  {
    Expression expr = MCUDAUtils.getBidxID(i);
    VariableDeclaration countDecl = (VariableDeclaration)body.findSymbol(getBidxCounterID());
    IRTools.replaceAll((Traversable)expr, MCUDAUtils.getBidxID().get(0), new Identifier((VariableDeclarator)countDecl.getDeclarator(0)));
    return expr;
  }

  private void createBlockLoopNest(Procedure proc)
  {
    proc.addDeclaration(MCUDAUtils.getGdimDecl().get(0));
    proc.addDeclaration(MCUDAUtils.getBdimDecl().get(0));

    if (Driver.getOptionValue("param_core") != null) {
      VariableDeclaration numCoresDecl = new VariableDeclaration(new UserSpecifier(new NameID("int")),
          new VariableDeclarator(getNumCoresID()));
      VariableDeclaration coreIDDecl = new VariableDeclaration(new UserSpecifier(new NameID("int")),
          new VariableDeclarator(getCoreID()));

      proc.addDeclaration(numCoresDecl);
      proc.addDeclaration(coreIDDecl);
    }
    CompoundStatement body = proc.getBody();
    VariableDeclaration vDecl = getBidxCounterDecl();

    LinkedList<Statement> detachStmts = new LinkedList();
    detachStmts.clear();

    FlatIterator flatIter = new FlatIterator(body);
    while (flatIter.hasNext()) {
      Statement currStmt = (Statement)flatIter.next();
      if (currStmt instanceof DeclarationStatement)
        continue;
      else
        detachStmts.add(currStmt);
    }

    CompoundStatement loopBody = new CompoundStatement();
    for (Statement currStmt : detachStmts) {
      currStmt.detach();
      loopBody.addStatement(currStmt);
    }
    detachStmts.clear();

    Statement bodyStmt = loopBody;
    ExpressionStatement exprStmt_y = new ExpressionStatement(new AssignmentExpression(
          MCUDAUtils.getBidxID(1),
          AssignmentOperator.NORMAL,
          new IntegerLiteral(0)));

    ExpressionStatement exprStmt_x;
    if (Driver.getOptionValue("param_core") != null) {
      exprStmt_x = new ExpressionStatement(new AssignmentExpression(
            MCUDAUtils.getBidxID(0),
            AssignmentOperator.NORMAL,
            getCoreID()));
    } else {
      exprStmt_x = new ExpressionStatement(new AssignmentExpression(
            MCUDAUtils.getBidxID(0),
            AssignmentOperator.NORMAL,
            new IntegerLiteral(0)));
    }
    FCUDAGlobalData.setInitializationBlkXStmt(proc, exprStmt_x);

    Expression condition_y = new BinaryExpression(
        MCUDAUtils.getBidxID(1), 
        BinaryOperator.COMPARE_GE,
        MCUDAUtils.getGdimID(1));
    Expression condition_x = new BinaryExpression(
        MCUDAUtils.getBidxID(0), 
        BinaryOperator.COMPARE_GE,
        MCUDAUtils.getGdimID(0));

    CompoundStatement bodyCheckBlkX = new CompoundStatement();
    ExpressionStatement reduceBlkX = new ExpressionStatement(new AssignmentExpression(
          MCUDAUtils.getBidxID(0),
          AssignmentOperator.NORMAL,
          new BinaryExpression(
            MCUDAUtils.getBidxID(0),
            BinaryOperator.SUBTRACT,
            MCUDAUtils.getGdimID(0))));
    ExpressionStatement increaseBlkY = new ExpressionStatement(new AssignmentExpression(
          MCUDAUtils.getBidxID(1),
          AssignmentOperator.NORMAL,
          new BinaryExpression(
            MCUDAUtils.getBidxID(1),
            BinaryOperator.ADD,
            new IntegerLiteral(1))));

    ExpressionStatement increaseBlkX;
    if (Driver.getOptionValue("param_core") != null) {
      increaseBlkX = new ExpressionStatement(new AssignmentExpression(
            MCUDAUtils.getBidxID(0),
            AssignmentOperator.NORMAL,
            new BinaryExpression(
              MCUDAUtils.getBidxID(0),
              BinaryOperator.ADD,
              getNumCoresID())));
    } else {
      increaseBlkX = new ExpressionStatement(new AssignmentExpression(
            MCUDAUtils.getBidxID(0),
            AssignmentOperator.NORMAL,
            new BinaryExpression(
              MCUDAUtils.getBidxID(0),
              BinaryOperator.ADD,
              new IntegerLiteral(1))));
    }
    FCUDAGlobalData.setIncrementBlkXStmt(proc, increaseBlkX);

    bodyCheckBlkX.addStatement(reduceBlkX);
    bodyCheckBlkX.addStatement(increaseBlkY);
    WhileLoop whileCheckBlkX = new WhileLoop(
        condition_x, 
        bodyCheckBlkX);

    FCUDAGlobalData.setBlkXLoop(proc, whileCheckBlkX);

    IfStatement ifCheckBlkY = new IfStatement(
        condition_y,
        new BreakStatement());
    FCUDAGlobalData.setBlkYIf(proc, ifCheckBlkY);

    CompoundStatement currBody = new CompoundStatement();
    currBody.addStatement(whileCheckBlkX);
    currBody.addStatement(ifCheckBlkY);
    currBody.addStatement(bodyStmt);
    currBody.addStatement(increaseBlkX);
    WhileLoop whileBlk = new WhileLoop(
        new IntegerLiteral(1),
        currBody);

    FCUDAGlobalData.setBlockIdxLoop(proc, whileBlk);

    CompoundStatement blkIdxHandle = new CompoundStatement();
    blkIdxHandle.addStatement(exprStmt_y);
    blkIdxHandle.addStatement(exprStmt_x);
    blkIdxHandle.addStatement(whileBlk);
    bodyStmt = blkIdxHandle;
    body.addStatement(bodyStmt);

  }

  private void flattenUnnecessaryCompoundStmts(CompoundStatement cStmt)
  {
    flattenUnnecessaryCompoundStmts(cStmt, null);
  }

  //Removes unnecesary child CompoundStatements 
  private void flattenUnnecessaryCompoundStmts(CompoundStatement cStmt, LinkedList<Statement> addedStmts)
  {
    HashMap<CompoundStatement, LinkedList<Statement>> flattenChildren = new HashMap();
    flattenChildren.clear();

    FlatIterator flatIter = new FlatIterator(cStmt);
    while (flatIter.hasNext()) {
      Statement currStmt = null;
      try {
        currStmt = (Statement)flatIter.next(Statement.class);
      }
      catch(NoSuchElementException e) {
        break;
      }

      if (currStmt instanceof CompoundStatement) {
        CompoundStatement currCStmt = (CompoundStatement)currStmt;
        LinkedList<Statement> tmpList = new LinkedList();
        flattenUnnecessaryCompoundStmts(currCStmt, tmpList);
        flattenChildren.put(currCStmt, tmpList);
        continue;
      }
      if (addedStmts != null)
        addedStmts.add(currStmt);
      if (currStmt instanceof ForLoop)
        flattenUnnecessaryCompoundStmts((CompoundStatement)((ForLoop)currStmt).getBody());
      if (currStmt instanceof WhileLoop)
        flattenUnnecessaryCompoundStmts((CompoundStatement)((WhileLoop)currStmt).getBody());
      if (currStmt instanceof IfStatement) {
        flattenUnnecessaryCompoundStmts((CompoundStatement)((IfStatement)currStmt).getThenStatement());
        if (((IfStatement)currStmt).getElseStatement() != null)
          flattenUnnecessaryCompoundStmts((CompoundStatement)((IfStatement)currStmt).getElseStatement());
      }
    }

    for (Map.Entry<CompoundStatement, LinkedList<Statement>> currEntry : flattenChildren.entrySet()) {
      Statement lastStmt = currEntry.getKey();
      for (Statement currStmt : currEntry.getValue()) {
        currStmt.detach();
        cStmt.addStatementAfter(lastStmt, currStmt);
        lastStmt = currStmt;
      }
      (currEntry.getKey()).detach();
    }

    flattenChildren.clear();
  }

  public void transformProcedure(Procedure proc)
  {
    if (!(FCUDAGlobalData.isConstMemProc(proc))) {
      mProcedure = proc;
      createBlockLoopNest(proc);
      flattenUnnecessaryCompoundStmts(proc.getBody(), null);
    }
  }
}
