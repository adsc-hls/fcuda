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

import fcuda.ir.*;
import fcuda.utils.*;

import cetus.exec.*;
import cetus.hir.*;

/**
 * Interfaces a function call to the MCUDA runtime library.
 * First creates a type definition for this function's parameter 
 * structure.  
 * Then modifies the current function to now accept the implicit 
 * block and grid variables of the CUDA programming model, 
 * in addition to a pointer to an instance of the parameter structure.
 * Lastly, creates a new function with the original function 
 * signature, whose sole purpose is to construct the paramter 
 * structure, and pass the kernel invocation to the runtime.  
 * The kernel function is renamed and passed as one of the 
 * kernel invocation parameters.  
 */
public class WrapDeviceParams extends KernelTransformPass
{
  //Some commonly used names and structures across functions
  String kernelName;
  String paramStructName;
  //*cetus-1.1*  Identifier paramStructID;
  IDExpression paramStructID;
  List<VariableDeclaration> realParamList;
  ClassDeclaration paramStruct;

  public String getPassName()
  {
    return new String("[WrapDeviceParams-MCUDA]");
  }

  public WrapDeviceParams(Program program)
  {
    super(program);
  }

  //*cetus-1.1*  public Identifier kernelIdentifier()
  public IDExpression kernelIdentifier()
  {
    //*cetus-1.1*   return new Identifier(new String(kernelName));
    return new NameID(new String(kernelName));
  }

  private UserSpecifier paramStructType()
  {
    //*cetus-1.1*  return new UserSpecifier((Identifier)paramStructID.clone());
    return new UserSpecifier((IDExpression)paramStructID.clone());
  }

  private List<VariableDeclaration> implicitParams()
  {
    List<VariableDeclaration> ret = MCUDAUtils.Gdim.getHostDecl();
    ret.addAll(MCUDAUtils.Bdim.getHostDecl());
    return ret;
  } 

  private Expression buildCEANTidAccess(int i)
  {
    Expression result = MCUDAUtils.Tidx.getId(i);

    if(Driver.getOptionValue("CEANv2") != null)
    {
      for(Expression e : MCUDAUtils.getBdim())
        result = new ArrayAccess(result, new ArraySlice());
    } else
      result = new ArrayAccess(result, new ArraySlice());

    return result;
  }

  private CompoundStatement generateNolibInterfaceBody()
  {
    // Create new interface function body
    CompoundStatement interfaceBody = new CompoundStatement();

    //Just create a blockIdx variable, for some for loops
    List<VariableDeclaration> blockIdxDecl = MCUDAUtils.Bidx.getHostDecl();

    for(VariableDeclaration bd : blockIdxDecl)
      interfaceBody.addDeclaration(bd);

    if(Driver.getOptionValue("Xpilot") == null)
    {
      Expression bidz = MCUDAUtils.Bidx.getHostId(2);

      AssignmentExpression clearBlockz = 
        new AssignmentExpression(bidz, AssignmentOperator.NORMAL, 
            new IntegerLiteral(0));

      ExpressionStatement clearZStmt = 
        new ExpressionStatement(clearBlockz);

      interfaceBody.addStatement(clearZStmt);
    }

    //Add initializers for the Tidx arrays if running on CEAN
    if(Driver.getOptionValue("CEAN") != null)
    {
      List<VariableDeclaration> tidDecl = MCUDAUtils.Tidx.getDecl();
      for(VariableDeclaration currentDecl : tidDecl)
      {
        if(Driver.getOptionValue("CEANv2") != null)
        {
          List<Expression> bdim_size = MCUDAUtils.getBdim();
          for(int i = bdim_size.size() - 1; i >= 0; i--)
            currentDecl.getDeclarator(0).getArraySpecifiers().add(0, new ArraySpecifier(bdim_size.get(i)));
        } else
          currentDecl.getDeclarator(0).getArraySpecifiers().add(0, new ArraySpecifier(MCUDAUtils.getMaxNumThreads()));
        interfaceBody.addDeclaration(currentDecl);
      }

      List<Expression> tidValue = new LinkedList<Expression>();

      if(Driver.getOptionValue("CEANv2") != null)
      {
        int localdim = 0;
        for(int i = 0; i < MCUDAUtils.Bdim.getNumEntries(); i++)
        {
          if(MCUDAUtils.getBdim(i) > 1)
            //*cetus-1.1*  tidValue.add(new Identifier("__sec_implicit_index(" + localdim++ + ")"));
            tidValue.add(new NameID("__sec_implicit_index(" + localdim++ + ")"));
          else
            tidValue.add(new IntegerLiteral(0));
        }
      } 
      else
      {
        //Okay, so we're cheating.  So what?
        //*cetus-1.1*  tidValue.add(new Identifier("__sec_implicit_index(0)%" + MCUDAUtils.Bdim.getHostId(0).toString()));
        tidValue.add(new NameID("__sec_implicit_index(0)%" + MCUDAUtils.Bdim.getHostId(0).toString()));
        //*cetus-1.1*  tidValue.add(new Identifier("(__sec_implicit_index(0)/" + MCUDAUtils.Bdim.getHostId(0).toString() + ")%" + 
        tidValue.add(new NameID("(__sec_implicit_index(0)/" + MCUDAUtils.Bdim.getHostId(0).toString() + ")%" + 
              MCUDAUtils.Bdim.getHostId(1).toString()));
        //*cetus-1.1*   tidValue.add(new Identifier("__sec_implicit_index(0)/(" + MCUDAUtils.Bdim.getHostId(0).toString() + "*" + 
        tidValue.add(new NameID("__sec_implicit_index(0)/(" + MCUDAUtils.Bdim.getHostId(0).toString() + "*" + 
              MCUDAUtils.Bdim.getHostId(1).toString() + ")"));
      }


      for(int i = 0; i < tidValue.size(); i++)
      {
        Expression assign = new AssignmentExpression( buildCEANTidAccess(i), 
            AssignmentOperator.NORMAL, 
            tidValue.get(i));
        interfaceBody.addStatement(new ExpressionStatement(assign));
      }
    }

    //Create the function call
    FunctionCall kernelCall = new FunctionCall(kernelIdentifier());

    for( VariableDeclaration pdecl : realParamList )
    {
      VariableDeclarator pDeclarator = 
        (VariableDeclarator)(pdecl.getDeclarator(0));

      //*cetus-1.1*   Identifier id = (Identifier) pDeclarator.getDirectDeclarator().clone();
      Identifier id = (Identifier) pDeclarator.getID().clone();

      kernelCall.addArgument(id);
    }

    for(Expression e : MCUDAUtils.Bidx.getInterfaceId())
      kernelCall.addArgument(e);
    for(Expression e : MCUDAUtils.Bdim.getInterfaceId())
      kernelCall.addArgument(e);
    for(Expression e : MCUDAUtils.Gdim.getInterfaceId())
      kernelCall.addArgument(e);
    if(Driver.getOptionValue("CEAN") != null)
      for(Expression e : MCUDAUtils.Tidx.getId())
        kernelCall.addArgument(e);

    //Create body wrappy for kernel call
    ExpressionStatement kernelStmt = 
      new ExpressionStatement(kernelCall);

    Statement stmt = kernelStmt;

    if(Driver.getOptionValue("CEAN") != null)
    {
      CompoundStatement cstmt = new CompoundStatement();
      cstmt.addStatement(new AnnotationStatement(new PragmaAnnotation("noinline")));
      cstmt.addStatement(kernelStmt);
      stmt = cstmt;
    }


    for( int i = 0; i < 2 ; i++)
    {
      //Init
      Expression assignBidx = 
        new AssignmentExpression(MCUDAUtils.Bidx.getHostId(i), AssignmentOperator.NORMAL,
            new IntegerLiteral(0));

      ExpressionStatement assignBidxStmt = 
        new ExpressionStatement(assignBidx);

      //Comparison
      BinaryExpression BidxComp = 
        new BinaryExpression(MCUDAUtils.Bidx.getHostId(i), 
            BinaryOperator.COMPARE_LT, MCUDAUtils.Gdim.getHostId(i));

      //Update
      Expression bidxUpdate = 
        new UnaryExpression(UnaryOperator.POST_INCREMENT,
            MCUDAUtils.Bidx.getHostId(i));
      stmt = new ForLoop(assignBidxStmt, BidxComp, bidxUpdate, stmt);
    }

    interfaceBody.addStatement(stmt);

    return interfaceBody;
  }

  private CompoundStatement generateInterfaceBody()
  {
    // Create new interface function body
    CompoundStatement interfaceBody = new CompoundStatement();

    if(Driver.getOptionValue("staticKernelParams") == null) {
      // Instantiate a variable for the parameter structure
      List<Specifier> paramVarType = new LinkedList<Specifier>();

      paramVarType.add(paramStructType());
      paramVarType.add(PointerSpecifier.UNQUALIFIED);
      VariableDeclarator paramsDeclarator = 
        new VariableDeclarator(MCUDAUtils.getParamsID(kernelName));

      VariableDeclaration paramsDecl = 
        new VariableDeclaration(paramVarType, paramsDeclarator);

      interfaceBody.addDeclaration(paramsDecl);

      //Instantiate a malloc
      List<Specifier> mallocType = new LinkedList<Specifier>();
      //*cetus-1.1* mallocType.add(new UserSpecifier(new Identifier("struct")));
      mallocType.add(new UserSpecifier(new NameID("struct")));
      mallocType.add(paramStructType());

      SizeofExpression mallocArg = 
        new SizeofExpression(mallocType);
      //*cetus-1.1*  FunctionCall allocation = new FunctionCall(new Identifier("malloc"));
      FunctionCall allocation = new FunctionCall(new NameID("malloc"));
      allocation.addArgument(mallocArg);
      List<Specifier> specifierList = new LinkedList<Specifier>();
      specifierList.add(paramStructType());
      specifierList.add(PointerSpecifier.UNQUALIFIED);

      Typecast allocTypecast = new Typecast(specifierList, allocation);

      AssignmentExpression assignAlloc = 
        new AssignmentExpression(MCUDAUtils.getParamsID(kernelName),
            AssignmentOperator.NORMAL,
            allocTypecast);

      ExpressionStatement allocStmt = 
        new ExpressionStatement(assignAlloc);

      interfaceBody.addStatement(allocStmt);
    }

    //For each parameter, pack it in.

    for( VariableDeclaration d : realParamList )
    {
      VariableDeclarator pdeclName = (VariableDeclarator)d.getDeclarator(0);
      //*cetus-1.1*  Identifier pID = (Identifier) pdeclName.getDirectDeclarator();
      IDExpression pID = pdeclName.getID();
      //pID.setSymbol(((VariableDeclaration)paramStruct.findSymbol(pID)).getDeclarator(0));
      //Assign parameter value to structure
      AccessExpression paramField =
        new AccessExpression(MCUDAUtils.getParamsID(kernelName), 
            MCUDAUtils.paramsAccessOperator(), pID);

      //*cetus-1.1*   pID = (Identifier) pdeclName.getDirectDeclarator();
      //pID = (Identifier) pdeclName.getID();
      //*cetus-1.1*  pID.setSymbol(d.getDeclarator(0));

      AssignmentExpression assignParam = 
        new AssignmentExpression(paramField, AssignmentOperator.NORMAL, pID);

      ExpressionStatement assignParamStmt = 
        new ExpressionStatement(assignParam);

      interfaceBody.addStatement(assignParamStmt);
    }

    //For Rigel, we need to predicate parameter packing on being core 0,
    //  and barrier before launching the kernel
    if(Driver.getOptionValue("Rigel") != null || Driver.getOptionValue("RigelSP") != null)
    {
      //*cetus-1.1*  Identifier rigelCoreNum = new Identifier("rigel_thread_num");
      IDExpression rigelCoreNum = new NameID("rigel_thread_num");
      Declaration rigelCoreDecl = new VariableDeclaration(Specifier.INT, 
          new VariableDeclarator(rigelCoreNum));
      Statement getRigelCoreNum = new ExpressionStatement(
          new AssignmentExpression((Expression)rigelCoreNum.clone(), 
            AssignmentOperator.NORMAL, 
            //*cetus-1.1*   new FunctionCall(new Identifier("RigelGetThreadNum"))));
              new FunctionCall(new NameID("RigelGetThreadNum"))));
      BinaryExpression coreIDComp = 
        new BinaryExpression((Expression)rigelCoreNum.clone(), 
            BinaryOperator.COMPARE_EQ, new IntegerLiteral(0));
      Statement rigelPack = new IfStatement(coreIDComp, interfaceBody);
      /*
         FunctionCall rigelBarrier = 
         new FunctionCall(new Identifier("kernel_launch_barrier"));
         rigelBarrier.addArgument((Expression)rigelCoreNum.clone());
         Statement barrierStmt = new ExpressionStatement(rigelBarrier);
         */
      interfaceBody = new CompoundStatement();
      interfaceBody.addDeclaration(rigelCoreDecl);
      interfaceBody.addStatement(getRigelCoreNum);
      interfaceBody.addStatement(rigelPack);
      // Deprecated 12-09-2009
      //interfaceBody.addStatement(barrierStmt);
    }

    //Instantiate the kernel launch call
    FunctionCall kernelLaunch = 
      //*cetus-1.1*   new FunctionCall(new Identifier("__mcuda_kernel_launch"));
      new FunctionCall(new NameID("__mcuda_kernel_launch"));

    //Function pointer to kernel
    kernelLaunch.addArgument(kernelIdentifier());

    //Grid and Block dims
    for( Expression e : MCUDAUtils.Gdim.getInterfaceId() )
      kernelLaunch.addArgument(e);
    for( Expression e : MCUDAUtils.Bdim.getInterfaceId() )
      kernelLaunch.addArgument(e);

    //reference to pointer to parameters
    if(Driver.getOptionValue("staticKernelParams") == null) {
      UnaryExpression paramAddress = 
        new UnaryExpression(UnaryOperator.ADDRESS_OF,
            MCUDAUtils.getParamsID(kernelName));

      List<Specifier> voidPP = new ChainedList<Specifier>();
      voidPP.add(Specifier.VOID);
      voidPP.add(PointerSpecifier.UNQUALIFIED);
      voidPP.add(PointerSpecifier.UNQUALIFIED);

      Typecast voidPPTypecast = new Typecast(voidPP, paramAddress);

      kernelLaunch.addArgument(voidPPTypecast);
    }

    ExpressionStatement launchStmt = new ExpressionStatement(kernelLaunch);

    interfaceBody.addStatement(launchStmt);

    return interfaceBody;
  }

  public void transformProcedure(Procedure proc)
  {
    //Some commonly used names and structures
    kernelName = proc.getName().toString() + "_mc";
    String paramStructName = kernelName + "_paramstruct";
    //*cetus-1.1*  paramStructID = new Identifier(paramStructName);
    paramStructID = new NameID(paramStructName);

    TranslationUnit kernelFile = (TranslationUnit)proc.getParent();
    // New body for the kernel function
    CompoundStatement oldbody = proc.getBody();
    realParamList = new LinkedList<VariableDeclaration>();

    //If there are void parameters, filter them out.  
    //*cetus-1.1*   for( Declaration d : proc.getParameters() )
    for( Object d : proc.getParameters() )
    {
      if(d.toString().compareTo("void") != 0)
        realParamList.add((VariableDeclaration)d);
    }

    //If it's an empty function, don't even bother.  
    if(oldbody.getChildren().isEmpty())
      return;

    // Set up the structure for the parameters 
    if(Driver.getOptionValue("packedKernelParams") != null)
    {
      paramStruct = 
        new ClassDeclaration(ClassDeclaration.STRUCT, paramStructID);

      for( Declaration d : realParamList )
        paramStruct.addDeclaration(d);

      kernelFile.addDeclarationBefore(proc, paramStruct);

      MCUDAUtils.addStructTypedefBefore(kernelFile, proc, paramStruct);

      if(Driver.getOptionValue("staticKernelParams") != null)
      {
        VariableDeclarator paramsDecl = 
          new VariableDeclarator(MCUDAUtils.getParamsID(kernelName));
        VariableDeclaration paramsVar = 
          new VariableDeclaration(paramStructType(), paramsDecl);
        kernelFile.addDeclarationBefore(proc, paramsVar);
      }
    }


    //Generate the interface function's body
    CompoundStatement interfaceBody;
    if(Driver.getOptionValue("mcuda_nolib") == null) 
      interfaceBody = generateInterfaceBody();
    else
      interfaceBody = generateNolibInterfaceBody();

    proc.setBody(interfaceBody);

    //unpack the parameters on the kernel function side
    if(Driver.getOptionValue("packedKernelParams") != null)
    {
      Statement firstStmt = (Statement)oldbody.getChildren().get(0);
      Declaration firstDecl = null;
      if(firstStmt instanceof DeclarationStatement)
        firstDecl = ((DeclarationStatement)firstStmt).getDeclaration();

      // Add declarations for unpacking the parameters


      Expression paramsExpr;
      if(Driver.getOptionValue("staticKernelParams") == null)
      {
        List<Specifier> specifierList = new ChainedList<Specifier>();
        specifierList.add(paramStructType());
        specifierList.add(PointerSpecifier.UNQUALIFIED);
        paramsExpr = 
          new Typecast(specifierList, MCUDAUtils.getParamsID(kernelName));
      } else
        paramsExpr = MCUDAUtils.getParamsID(kernelName);

      for( VariableDeclaration pdecl : realParamList )
      {
        VariableDeclaration newDecl = (VariableDeclaration)pdecl.clone();

        VariableDeclarator pDeclarator = 
          (VariableDeclarator)(newDecl.getDeclarator(0));

        //*cetus-1.1*   Identifier id = (Identifier) pDeclarator.getDirectDeclarator();
        IDExpression id = pDeclarator.getID();
        //*cetus-1.1*   id.setSymbol(((VariableDeclaration)paramStruct.findSymbol(id)).getDeclarator(0));
        AccessExpression param_access = 
          new AccessExpression((Expression)paramsExpr.clone(), 
              MCUDAUtils.paramsAccessOperator(), id);

        pDeclarator.setInitializer(new Initializer(param_access));

        if(firstDecl == null)
          oldbody.addANSIDeclaration(newDecl);
        else
          oldbody.addDeclarationBefore(firstDecl, newDecl);
      }
    }

    //Get the parameter list for a kernel function 
    //(runtime implementation specified)
    List<Declaration> kernelParams = new LinkedList<Declaration>();

    if(Driver.getOptionValue("packedKernelParams") == null)
    { 
      for( VariableDeclaration d : realParamList )
        kernelParams.add((VariableDeclaration)d.clone());
    }
    kernelParams.addAll(MCUDAUtils.getKernelParams(kernelName));

    if(Driver.getOptionValue("CEAN") != null)
      for(VariableDeclaration tDecl : MCUDAUtils.Tidx.getDecl())
      {
        if(Driver.getOptionValue("CEANv2") != null)
        {
          List<Expression> bdim_size = MCUDAUtils.getBdim();
          for(int i = bdim_size.size() - 1; i >= 0; i--)
            tDecl.getDeclarator(0).getArraySpecifiers().add(0, new ArraySpecifier(bdim_size.get(i)));
        } else
          tDecl.getDeclarator(0).getArraySpecifiers().add(0, new ArraySpecifier(MCUDAUtils.getMaxNumThreads()));
        kernelParams.add(tDecl);
      }

    //Declare the new kernel
    ProcedureDeclarator newKernelDecl = 
      new ProcedureDeclarator( kernelIdentifier() , kernelParams);

    List<Specifier> newKernelSpecs = new ChainedList<Specifier>();
    newKernelSpecs.add(Specifier.GLOBAL);
    newKernelSpecs.add(Specifier.VOID);

    Procedure wrapped = new Procedure(newKernelSpecs, newKernelDecl, oldbody);
    kernelFile.addDeclarationBefore(proc, wrapped);

    //Add the implicit kernel function parameters to the now intermediate 
    // procedure
    for(VariableDeclaration d : implicitParams())
      proc.addDeclaration(d);

    //Don't mark the interface function as the kernel anymore
    proc.removeProcedureSpec(Specifier.GLOBAL);
  }

  public void transformPrototype(ProcedureDeclarator proto)
  {
    //Add the implicit kernel function parameters
    for(VariableDeclaration d : implicitParams())
      proto.addParameter(d);
  }

}
