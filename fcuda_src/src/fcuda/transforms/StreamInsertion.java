package fcuda.transforms;

import java.util.*;

import fcuda.common.*;
import fcuda.ir.*;
import fcuda.utils.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;


/* JAS: This class converts a program with a stream pragma into a new kernel instantiating the
 * constant memory buffer as a shared memory buffer, with a loop structure introduced in a
 * kernel caller function to process the entire off-chip array from which data is being streamed.
 * It also converts the stream pragma into a burst pragma for later phases to process correctly.
 *
 * Note that only one stream pragma is allowed per kernel.
 */

public class StreamInsertion extends KernelTransformPass
{
  private Procedure mProcedure;

  private List<VariableDeclaration> mSizeDecls;

  private List<Statement> mSizeDefs;

  private Set<IDExpression> mConstArrays;

  public String getPassName()
  {
    return new String("[StreamInsertion-FCUDA]");
  }

  public StreamInsertion(Program program)
  {
    super(program);
    mProcedure = null;
    mSizeDecls = new LinkedList<VariableDeclaration>();
    mSizeDefs = new LinkedList<Statement>();
    mConstArrays = new HashSet<IDExpression>();
    mConstArrays.clear();
  }

  // Find declaration and first definition of Variable (Assuming that stream pragma
  // is located at the beginning of kernel, thus first definition should be fine)
  // FIXME - Doesnt handle multi-statement definitions yet
  private void findSizeDef(IDExpression idExp)
  {
    boolean declFound = false;
    DepthFirstIterator iter = new DepthFirstIterator(mProcedure.getBody());

    Tools.printlnStatus("Looking for decl/def of "+idExp.toString(), 9);

    while (iter.hasNext()) {
      Statement stmt = null;

      try {
        stmt = (Statement)iter.next(Statement.class);
      } catch (NoSuchElementException e) {
        break;
      }

      if (stmt instanceof DeclarationStatement) {
        VariableDeclaration decl = (VariableDeclaration)((DeclarationStatement)stmt).getDeclaration();
        List<IDExpression> declIDs = decl.getDeclaredIDs();
        if (declIDs.contains(idExp)) {
          mSizeDecls.add((VariableDeclaration)decl.clone());
          declFound = true;
          continue;
        }
      }

      if (declFound && (stmt instanceof ExpressionStatement)) {
        Expression expr = ((ExpressionStatement)stmt).getExpression();
        Set<Expression> defSet = DataFlow.mayDefine(expr);
        if (defSet.contains(idExp)) {
          mSizeDefs.add((Statement)stmt.clone());
          break;
        }
      }
    }
  }

  /* Assumes that all stmts in parameter list are immediate descendants of body,
   * that streamBegin and streamEnd are as well, and that stmts does not contain
   * either streamBegin or streamEnd */
  private void createStreamLoopProc(CompoundStatement krnBody, AnnotationStatement streamBegin)
  {

    /* Collect the necessary information from the pragma for the loop */
    FcudaAnnotation fcAnnot = streamBegin.getAnnotations(FcudaAnnotation.class).get(0);
    LinkedList<String> mOnchipPtrNameList = new LinkedList();
    FCUDAutils.parseList((String)fcAnnot.get("local"), mOnchipPtrNameList);
    LinkedList<String> mOffchipPtrNameList = new LinkedList();
    FCUDAutils.parseList((String)fcAnnot.get("pointer"), mOffchipPtrNameList);
    LinkedList<String> sizeList = new LinkedList();
    FCUDAutils.parseList((String)fcAnnot.get("size"), sizeList);
    LinkedList<String> rsizeList = new LinkedList();
    FCUDAutils.parseList((String)fcAnnot.get("rsize"), rsizeList);
    String task = fcAnnot.get("name");

    CompoundStatement streamLoopBody = new CompoundStatement();
    CompoundStatement streamProcBody = new CompoundStatement();

    NameID streamLoopCounter = new NameID("fcStreamIter_" + task);

    VariableDeclarator lcName = new VariableDeclarator(streamLoopCounter);
    VariableDeclaration lcDecl = new VariableDeclaration(Specifier.INT, lcName);
    streamProcBody.addDeclaration(lcDecl);

    /* Create the stream loop */
    // *AP* FIXME - what about handling multi-dimensional CONSTANT Arrays??
    AssignmentExpression loopInit = new AssignmentExpression(new Identifier(lcName),
        AssignmentOperator.NORMAL,
        new IntegerLiteral(0));
    ExpressionStatement initStmt = new ExpressionStatement(loopInit);

    // Boundary Condition: Allow literal and variable rsize values
    Expression boundExp = null;
    IDExpression boundID = null;
    int sizeInt = 0;
    try {
      sizeInt = Integer.parseInt(rsizeList.get(0));
      boundExp = (Expression)(new IntegerLiteral(sizeInt));
    } catch(Exception e) {
      boundExp = new NameID(rsizeList.get(0));
      boundID = (IDExpression) boundExp;
      findSizeDef((IDExpression)boundExp);
    }

    BinaryExpression compare = new BinaryExpression(new Identifier(lcName),
        BinaryOperator.COMPARE_LT,
        boundExp);
    // Update Expression: Allow literal and variable size values
    Expression updateExp = null;
    IDExpression updateID = null;
    try {
      sizeInt = Integer.parseInt(sizeList.get(0));
      updateExp = (Expression)(new IntegerLiteral(sizeInt));
    }
    catch(Exception e)  {
      updateExp = (Expression)new NameID(sizeList.get(0));
      updateID = (IDExpression) updateExp;
      findSizeDef((IDExpression)updateExp);
    }

    AssignmentExpression update = new AssignmentExpression(new Identifier(lcName),
        AssignmentOperator.ADD,
        updateExp);

    ForLoop streamLoop = new ForLoop(initStmt, compare, update, streamLoopBody);


    /* Move the transfer pragma into the loop body */
    krnBody.removeChild(streamBegin);
    streamLoopBody.addStatement(streamBegin);

    // List to hold copies of constant memory declarations
    List<VariableDeclaration> constDecls = new LinkedList<VariableDeclaration>();
    constDecls.clear();

    List<Declaration> declList = new LinkedList<Declaration>();
    List<Declaration> declListCopy = new LinkedList<Declaration>();
    List<Expression> kernParams = new LinkedList<Expression>();
    // Create the shared memory array for the stream cache, and copy statement to it
    // Add off-chip pointer to kernel parameters
    for (int i = 0; i < mOnchipPtrNameList.size(); i++) {
      String var = mOnchipPtrNameList.get(i);
      NameID vID = new NameID(var);
      mConstArrays.add(vID);
      /* Find the original constant mem declaration  */
      VariableDeclaration constDecl = (VariableDeclaration) SymbolTools.findSymbol(krnBody, vID);
      VariableDeclarator constDeclarator = (VariableDeclarator) constDecl.getDeclarator(0);
      List<Specifier> vSpecs = (new ChainedList<Specifier>()).addLink(Specifier.SHARED);
      vSpecs.addAll(constDecl.getSpecifiers());
      List<Specifier> vArraySpecs = new LinkedList<Specifier>();
      for (Object aSpec : constDeclarator.getArraySpecifiers()) {
        if (!(aSpec instanceof ArraySpecifier))
          continue;
        for (int j = 0; j < ((ArraySpecifier)aSpec).getNumDimensions(); j++) {
          Expression e = (Expression) (((ArraySpecifier)aSpec).getDimension(j)).clone();
          vArraySpecs.add(new ArraySpecifier(e));
        }
      }
      VariableDeclarator vName = new VariableDeclarator(vID,vArraySpecs);
      VariableDeclaration vDecl = new VariableDeclaration(vSpecs, vName);
      constDecls.add(vDecl);


      /**TAN*
       * Find size of the stream array
       */
      List<Specifier> stream_vArraySpecs = new LinkedList<Specifier>();
      stream_vArraySpecs.add(new ArraySpecifier(boundExp));

      // Add off-chip pointer to kernel parameters
      NameID ptrID = new NameID(mOffchipPtrNameList.get(i));
      VariableDeclarator pDeclor =  new VariableDeclarator( PointerSpecifier.UNQUALIFIED, ptrID);
      vSpecs.remove(Specifier.CONSTANT);
      vSpecs.remove(Specifier.SHARED);
      VariableDeclaration pDecl = new VariableDeclaration(vSpecs, pDeclor);
      declListCopy.add(pDecl);

      // Add Constant mem declaration to kernel parameters
      mProcedure.addDeclaration((Declaration)vDecl.clone());

      /* Create a statement copying from the remote to local pointer */
      // **AP** FIXME - what about handling multi-dimensional CONSTANT Arrays??
      ArrayAccess dummyAcc = new ArrayAccess((Expression)vID.clone(), new IntegerLiteral(0));
      ArrayAccess remoteAcc = new ArrayAccess(new Identifier(pDeclor),
          new Identifier(lcName));
      Expression dummyExpr = new AssignmentExpression(dummyAcc, AssignmentOperator.NORMAL, remoteAcc);
      ExpressionStatement dummyStmt = new ExpressionStatement(dummyExpr);
      streamLoopBody.addStatement(dummyStmt);

      // Remove original CONSTANT declaration
      constDecl.detach();
    }

    /* Create the end transfer pragma */
    AnnotationStatement streamEnd = (AnnotationStatement)streamBegin.clone();
    streamLoopBody.addStatement(streamEnd);
    // Create kernel call and add to stream loop body
    FunctionCall kernCall = new FunctionCall((IDExpression) mProcedure.getName().clone());


    declList.addAll(mProcedure.getParameters());
    for (Declaration parDecl : declList) {
      VariableDeclarator parDeclor = (VariableDeclarator)((VariableDeclaration)parDecl).getDeclarator(0);
      IDExpression parameter = parDeclor.getID();
      if (!mConstArrays.contains(parameter))
        declListCopy.add((Declaration) parDecl.clone());
      kernParams.add((Expression) parameter.clone());
    }

    // Add BlockDim and GridDim parameters
    kernParams.add(MCUDAUtils.Gdim.getId().get(0));
    kernParams.add(MCUDAUtils.Bdim.getId().get(0));

    if (Driver.getOptionValue("param_core") != null) {
      kernParams.add(new NameID("num_cores"));
      kernParams.add(new NameID("core_id"));
    }

    kernCall.setArguments(kernParams);
    ExpressionStatement callStmt = new ExpressionStatement(kernCall);
    streamLoopBody.addStatement((Statement)callStmt);

    // Populate Stream Proc with statements
    FCUDAGlobalData.addIgnoreDuringDuplication(FCUDAutils.getClosestParentStmt(lcDecl));
    for (VariableDeclaration declCopy : mSizeDecls) {
      streamProcBody.addDeclaration(declCopy);
      FCUDAGlobalData.addIgnoreDuringDuplication(FCUDAutils.getClosestParentStmt(declCopy));
    }
    for (Statement defCopy : mSizeDefs) {
      streamProcBody.addStatement(defCopy);
      FCUDAGlobalData.addIgnoreDuringDuplication(defCopy);
    }
    for (VariableDeclaration cDecl : constDecls)
      streamProcBody.addANSIDeclaration(cDecl);
    streamProcBody.addStatement(streamLoop);


    List<Annotation> preAnnotList = mProcedure.getAnnotations();
    mProcedure.removeAnnotations();
    // Create Stream Procdedure
    ProcedureDeclarator streamProcDecl = new ProcedureDeclarator(new NameID(mProcedure.getName().toString() + "_Stream"),
        declListCopy);
    List<Specifier> strmProcSpecs = new LinkedList<Specifier>();
    strmProcSpecs.add(Specifier.GLOBAL);
    strmProcSpecs.add(Specifier.VOID);
    Procedure streamProc = new Procedure(strmProcSpecs, streamProcDecl, streamProcBody);
    streamProc.addDeclaration(MCUDAUtils.getGdimDecl().get(0));
    streamProc.addDeclaration(MCUDAUtils.getBdimDecl().get(0));
    for (Annotation annot : preAnnotList) {
      if (annot.get("x_dim") != null) {
        mProcedure.annotate(annot);
      }
      else
        streamProc.annotate(annot);
    }
    FCUDAGlobalData.addConstMemProc(streamProc, mProcedure, streamLoop, streamLoopCounter, boundID, updateID);

    TranslationUnit filePrnt = (TranslationUnit)mProcedure.getParent();
    filePrnt.addDeclarationAfter(mProcedure, streamProc);

    // Store Constant arrays set
    FCUDAGlobalData.addConstMems(mConstArrays);


    if (Driver.getOptionValue("param_core") != null) {
      VariableDeclaration numCoresDecl = new VariableDeclaration(new UserSpecifier(new NameID("int")),
          new VariableDeclarator(new NameID("num_cores")));
      VariableDeclaration coreIDDecl = new VariableDeclaration(new UserSpecifier(new NameID("int")),
          new VariableDeclarator(new NameID("core_id")));

      streamProc.addDeclaration(numCoresDecl);
      streamProc.addDeclaration(coreIDDecl);
    }

    // Edit Pragmas
    fcAnnot.put("type", "stream");
    fcAnnot = streamEnd.getAnnotations(FcudaAnnotation.class).get(0);
    fcAnnot.put("type", "stream");
    fcAnnot.put("begin", "false");
    fcAnnot.put("end", "true");
  }


  // *AP* Assumption: Only one Stream annotation supported currently
  private void findStream(CompoundStatement krnbody)
  {
    FlatIterator iter = new FlatIterator(krnbody);
    AnnotationStatement fcStreamBegin = null;
    FcudaAnnotation fcAnnot;
    String annotType;
    boolean streamStatement = false;
    while (iter.hasNext()) {
      Statement stmt = null;
      try {
        stmt = (Statement)iter.next(Statement.class);
      } catch (NoSuchElementException e) {
        break;
      }

      if (!(stmt instanceof AnnotationStatement))
        continue;

      // Only interested in STREAM FCUDA pragmas
      List<FcudaAnnotation> fcAnnots = stmt.getAnnotations(FcudaAnnotation.class);
      if(fcAnnots.isEmpty())
        continue;

      fcAnnot = fcAnnots.get(0);

      annotType = (String) fcAnnot.get("fcuda");

      // **JAS** handle constant memory transfers
      if (!annotType.equals("transfer") || !fcAnnot.get("type").equals("stream"))
        continue;

      if (((String)fcAnnot.get("begin")).equals("true")) {
        streamStatement = true;
        fcStreamBegin = (AnnotationStatement)stmt;
        break;
      }
    }

    /* Only attempt if we actually found a stream transfer pragma */
    if (streamStatement == true)
      createStreamLoopProc(krnbody, fcStreamBegin);
  }

  public void transformProcedure(Procedure proc)
  {
    mProcedure = proc;
    findStream(proc.getBody());
  }
}


