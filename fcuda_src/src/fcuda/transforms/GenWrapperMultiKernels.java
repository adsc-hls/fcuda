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
import java.util.*;
import fcuda.utils.*;
import fcuda.ir.*;
import fcuda.analysis.*;
import fcuda.common.*;
import cetus.analysis.*;
import cetus.hir.*;
import cetus.transforms.*;
import cetus.exec.*;

/**
 * Generate wrapper of cores
 */

public class GenWrapperMultiKernels extends KernelTransformPass
{
  private Procedure mProcedure;
  private int numCores;
  private String wrapperSingleKernName;
  private static boolean wrapperMultiKernCreated = false;

  public String getPassName()
  {
    return new String("[GenWrapperMultiKernels - FCUDA]");
  }

  public GenWrapperMultiKernels(Program program)
  {
    super(program);
  }

  public void createWrapperMultiKernProc(Procedure proc)
  {
    LinkedList<PragmaAnnotation> listHLSPragmas = new LinkedList<PragmaAnnotation>();
    String scalarHLSPragma;
    List<Procedure> listWrappers = FCUDAGlobalData.getListWrapperSingleKern();
    List<Declaration> declList = new LinkedList<Declaration>();
    List<Declaration> declListCopy = new LinkedList<Declaration>();
    List<Expression> wrapperSingleKernParams = new LinkedList<Expression>();
    List<String> paramList = new LinkedList<String>();

    List<Specifier> wrapperMultiKernProcSpecs = new LinkedList<Specifier>();
    wrapperMultiKernProcSpecs.add(Specifier.GLOBAL);
    wrapperMultiKernProcSpecs.add(Specifier.VOID);
    CompoundStatement wrapperMultiKernProcBody = new CompoundStatement();


    for (Procedure wrapper : listWrappers) {
      wrapperSingleKernParams.clear();
      declList.clear();
      declList.addAll(wrapper.getParameters());

      for (Declaration parDecl : declList) {
        VariableDeclarator parDeclor = (VariableDeclarator)((VariableDeclaration)parDecl).getDeclarator(0);
        IDExpression parameter = parDeclor.getID();
        if (parameter.toString().contains("memport_core")) {
          String paramSpec = ((VariableDeclaration)parDecl).getSpecifiers().get(0).toString();
          NameID paramPortCore = new NameID(parameter.toString() + "_" + paramSpec);
          VariableDeclarator portCoreDeclor = new VariableDeclarator(
              parDeclor.getSpecifiers(),
              paramPortCore);
          VariableDeclaration portCoreDecl = new VariableDeclaration(
              ((VariableDeclaration)parDecl).getSpecifiers(),
              portCoreDeclor);

          if (!paramList.contains(portCoreDecl.toString())) {
            paramList.add(portCoreDecl.toString());
            declListCopy.add(portCoreDecl);
          }
          wrapperSingleKernParams.add(paramPortCore.clone());
        } else {
          if (!paramList.contains(parameter.toString())) {
            paramList.add(parameter.toString());
            declListCopy.add((Declaration) parDecl.clone());
          }
          wrapperSingleKernParams.add(parameter.clone());
        }
      }

      NameID enableWrapperID = new NameID("en_" + wrapper.getName().toString());
      VariableDeclarator enableWrapperDeclor = new VariableDeclarator(enableWrapperID);
      VariableDeclaration enableWrapperDecl = new VariableDeclaration(
          Specifier.INT, enableWrapperDeclor);
      declListCopy.add(enableWrapperDecl);
      scalarHLSPragma = new String("HLS INTERFACE ap_none register port=" + enableWrapperID.toString());
      listHLSPragmas.add(new PragmaAnnotation(scalarHLSPragma));
      scalarHLSPragma = new String("HLS RESOURCE core=AXI4LiteS variable=" + enableWrapperID.toString());
      listHLSPragmas.add(new PragmaAnnotation(scalarHLSPragma));

      Expression condCheckEnableWrapper = new BinaryExpression(
          enableWrapperID.clone(), 
          BinaryOperator.COMPARE_EQ,
          new IntegerLiteral(1));

      FunctionCall wrapperSingleKernCall = new FunctionCall((IDExpression) wrapper.getName().clone());
      wrapperSingleKernCall.setArguments(wrapperSingleKernParams);

      ExpressionStatement callStmt = new ExpressionStatement(wrapperSingleKernCall);

      IfStatement ifCheckEnableWrapper = new IfStatement(
          condCheckEnableWrapper,
          callStmt);

      wrapperMultiKernProcBody.addStatement((Statement)ifCheckEnableWrapper);
    }

    ProcedureDeclarator wrapperMultiKernProcDecl = new ProcedureDeclarator(
        new NameID("fcuda"),
        declListCopy);

    Procedure wrapperMultiKernProc = new Procedure(
        wrapperMultiKernProcSpecs, wrapperMultiKernProcDecl, wrapperMultiKernProcBody);

    scalarHLSPragma = new String("HLS RESOURCE core=AXI4LiteS variable=return");
    listHLSPragmas.add(new PragmaAnnotation(scalarHLSPragma));

    TranslationUnit filePrnt = (TranslationUnit)proc.getParent();
    filePrnt.addDeclaration(wrapperMultiKernProc);
    FCUDAGlobalData.addListHLSPragmas(wrapperMultiKernProc, listHLSPragmas);

    wrapperMultiKernCreated = true;
  }

  public void transformProcedure(Procedure proc)
  {
    if (wrapperMultiKernCreated)
      return;

    createWrapperMultiKernProc(proc);
  }
}
