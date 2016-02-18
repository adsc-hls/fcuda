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

public class GenWrapperSingleKernel extends KernelTransformPass
{
  public static int kernelNo = 1;

  public String getPassName()
  {
    return new String("[GenWrapperSingleKernel - FCUDA]");
  }

  public GenWrapperSingleKernel(Program program)
  {
    super(program);
  }

  public void createWrapperProc(Procedure proc)
  {
    LinkedList<PragmaAnnotation> listHLSPragmas = new LinkedList<PragmaAnnotation>();
    String scalarHLSPragma;
    List<Annotation> mergeAnnotations  = new java.util.ArrayList<Annotation>();
    HashMap<String, String> mergePorts = new HashMap<String, String>();
    List<String> listHLSDataPackPorts = new LinkedList<String>();
    int numCores = 1;
    boolean hasCoreInfoPragma = false;
    for (Annotation annot : proc.getAnnotations()) {
      if (annot.get("fcuda") == "portmerge") {
        mergePorts.put(annot.get("remove_port_name").toString(),
            annot.get("port_id").toString());
        mergeAnnotations.add(annot);
        if (annot.get("data_pack") != null && 
            annot.get("data_pack").toString().equals("yes"))
          listHLSDataPackPorts.add(
              annot.get("remove_port_name").toString());
      }

      if (annot.get("fcuda") == "coreinfo") {
        String numCoresStr = annot.get("num_cores");
        if (numCoresStr != null)
          numCores = Integer.parseInt(numCoresStr);
      }
    }

    for (Annotation annot : mergeAnnotations)
      annot.detach();

    mergeAnnotations.clear();

    List<Declaration> declList = new LinkedList<Declaration>();
    List<Declaration> declListCopy = new LinkedList<Declaration>();
    List<Expression> kernParams = new LinkedList<Expression>();

    declList.addAll(proc.getParameters());
    for (Declaration parDecl : declList) {
      VariableDeclarator parDeclor = (VariableDeclarator)((VariableDeclaration)parDecl).getDeclarator(0);
      IDExpression parameter = parDeclor.getID();
      if (!parameter.toString().equals("num_cores") &&
          !parameter.toString().equals("core_id")) {

        if (mergePorts.containsKey(parameter.toString())) {
          NameID paramAddress = new NameID(parameter.toString() + "_addr");
          VariableDeclarator addressDeclor = new VariableDeclarator(paramAddress);
          VariableDeclaration addressDecl = new VariableDeclaration(
              Specifier.INT, addressDeclor);
          declListCopy.add(addressDecl);
          scalarHLSPragma = new String("HLS INTERFACE ap_none register port=" + paramAddress.toString());
          listHLSPragmas.add(new PragmaAnnotation(scalarHLSPragma));
          scalarHLSPragma = new String("HLS RESOURCE core=AXI4LiteS variable=" + paramAddress.toString());
          listHLSPragmas.add(new PragmaAnnotation(scalarHLSPragma));
          
          for (int i = 0; i <numCores; i++) {
            NameID paramPortCore = new NameID(
                parameter.toString() + "_core" + Integer.toString(i));
            VariableDeclarator portCoreDeclor = new VariableDeclarator(
                parDeclor.getSpecifiers(),
                paramPortCore);
            VariableDeclaration portCoreDecl = new VariableDeclaration(
                ((VariableDeclaration)parDecl).getSpecifiers(),
                portCoreDeclor);
            declListCopy.add(portCoreDecl);

            Annotation annot = new FcudaAnnotation("fcuda", "portmerge");
            annot.put("remove_port_name", paramPortCore.toString());
            annot.put("port_id", mergePorts.get(parameter.toString()));
            annot.put("port_core", Integer.toString(i));
            annot.put("offset", paramAddress.toString());
            if (listHLSDataPackPorts.contains(parameter.toString())) {
              annot.put("data_pack", "yes");
            } else
              annot.put("data_pack", "no");
            mergeAnnotations.add(annot);
          }
        } else {
          declListCopy.add((Declaration) parDecl.clone());

          scalarHLSPragma = new String("HLS INTERFACE ap_none register port=" + parameter.toString());
          listHLSPragmas.add(new PragmaAnnotation(scalarHLSPragma));
          scalarHLSPragma = new String("HLS RESOURCE core=AXI4LiteS variable=" + parameter.toString());
          listHLSPragmas.add(new PragmaAnnotation(scalarHLSPragma));
        }
        kernParams.add((Expression) parameter.clone());
      }
    }

    scalarHLSPragma = new String("HLS RESOURCE core=AXI4LiteS variable=return");
    listHLSPragmas.add(new PragmaAnnotation(scalarHLSPragma));

    ProcedureDeclarator wrapperProcDecl = new ProcedureDeclarator(
        new NameID("fcuda" + kernelNo),
        declListCopy);
    List<Specifier> wrapperProcSpecs = new LinkedList<Specifier>();
    wrapperProcSpecs.add(Specifier.GLOBAL);
    wrapperProcSpecs.add(Specifier.VOID);
    CompoundStatement wrapperProcBody = new CompoundStatement();

    for (int i = 0; i < numCores; i++) {
      List<Expression> kernParamsCore = new LinkedList<Expression>();
      for (Expression param : kernParams) {
        if (mergePorts.containsKey(param.toString())) {
          IDExpression paramWithCoreID = new NameID(param.toString() + "_core" + Integer.toString(i));
          kernParamsCore.add(paramWithCoreID);
        } else
          kernParamsCore.add(param.clone());
      }
      kernParamsCore.add(new IntegerLiteral(numCores));
      kernParamsCore.add(new IntegerLiteral(i));
      FunctionCall kernCall = new FunctionCall((IDExpression) proc.getName().clone());
      kernCall.setArguments(kernParamsCore);

      ExpressionStatement callStmt = new ExpressionStatement(kernCall);
      wrapperProcBody.addStatement((Statement)callStmt);
    }
    Procedure wrapperProc = new Procedure(wrapperProcSpecs, wrapperProcDecl, wrapperProcBody);

    for (Annotation annot : mergeAnnotations) {
      wrapperProc.annotate(annot);
    }
    FCUDAGlobalData.addListHLSPragmas(wrapperProc, listHLSPragmas);
    TranslationUnit filePrnt = (TranslationUnit)proc.getParent();
    filePrnt.addDeclarationAfter(proc, wrapperProc);

    FCUDAGlobalData.addWrapperSingleKern(wrapperProc);
  }

  public void transformProcedure(Procedure proc)
  {
    // Check whether the current procedure is a child
    // of a stream procedure. If it is, we do not need
    // to generate a wrapper function for this procedure.
    Expression constMemKern = FCUDAGlobalData.getConstMemKern();
    if (((constMemKern != null) && 
          (!proc.getName().toString().equals(constMemKern.toString()))) || constMemKern == null) {
      createWrapperProc(proc);
      kernelNo = kernelNo + 1;
    }
  }
}
