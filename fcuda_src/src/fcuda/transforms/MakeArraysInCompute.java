package fcuda.transforms;

import java.util.*;

import fcuda.common.*;
import fcuda.utils.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;

/*
 * Use DataFlow analysis to find all scalar variables defined between compute begin and compute end pragmas
 * and used outside the compute section. All such variables must be converted to arrays based on ty,tx
 * Ignore variables that are pure functions of tx, ty, bx, by
 *
 */

public class MakeArraysInCompute extends KernelTransformPass
{
  private HashSet<Statement> mComputeStatements;
  private HashSet<Expression> mDefUseSet;
  private HashSet<VariableDeclarator> mVarsToConvert;
  private Procedure mProcedure;

  public String getPassName()
  {
    return new String("[MakeArraysInCompute-FCUDA]");
  }

  public MakeArraysInCompute(Program program)
  {
    super(program);
  }

  private void findVarsToConvert()
  {
    mDefUseSet.clear();
    for(Statement currStmt : mComputeStatements) {
      // *FIXME* Now only looks for RAW dependencies where the COMPUTE does the Write
      //         We should also look for RAW where the COMPUTE does the Read
      mDefUseSet.addAll(DataFlowTools.getDefSet(currStmt));
    }

    for(Expression currExpr : mDefUseSet) {
      //Ignore array accesses (arrays don't need to be converted)
      if(currExpr instanceof ArrayAccess)
        continue;

      //TODO - check if function of dim3 variables only
      // **AP** Should only make array of variables whose definition depends on threadIdx
      //        or variables that do not have different value for different threadIdx id

      Expression cleanExpr = null;

      // Handle structure accesses
      if (currExpr instanceof AccessExpression) {
        cleanExpr = ((BinaryExpression) currExpr).getLHS();
        System.out.println("WARNING: currently do not handle RHS elements of AccessExpression "+
            cleanExpr.toString()+ " in findVarsToConvert()");
      } else
        cleanExpr = currExpr;

      //If ID expression
      if (cleanExpr instanceof IDExpression) {
        boolean scalar, used;
        IDExpression idExpr = (IDExpression)cleanExpr;
        System.out.println("IDEXPR "+idExpr.toString());
        VariableDeclarator varDecl = FCUDAutils.getVariableDeclarator(idExpr);

        //Check if scalar and used out of COMPUTE
        scalar = FCUDAutils.isScalar(varDecl);
        used =  isDefUsedOutsideCompute(varDecl);
        System.out.println("[MakeArrays]: "+cleanExpr.toString()+" is scalar:"+scalar +" used:"+used);
        if (scalar && used)
          mVarsToConvert.add(varDecl);
      } else
        System.out.println("[MakeArraysInCompute]: "+ cleanExpr.toString()+"  expression not ID");
    }

    for (VariableDeclarator currDecl : mVarsToConvert)
      System.out.println("Variable to convert: "+currDecl.toString());
  } // findVarsToConvert()


  private boolean isDefUsedOutsideCompute(VariableDeclarator varDecl)
  {
    DepthFirstIterator iter = new DepthFirstIterator(mProcedure);
    iter.pruneOn(Expression.class);

    boolean storeStatements = false;
    HashSet<Expression> currDefUseSet = new HashSet();
    boolean returnVal = false;
    while (iter.hasNext()) {
      Expression currExpr = null;

      try {
        currExpr = (Expression)iter.next(Expression.class);
      } catch (NoSuchElementException e) {
        break;
      }

      //FIXME - assume all actual compute functions are inlined - ignore functions such as __syncthreads()
      if (currExpr instanceof FunctionCall)
        continue;

      Statement closestStmt = IRTools.getAncestorOfType(currExpr, Statement.class);

      //If declaration statement don't bother - needs ANSI decl pass to be run beforehand
      if (closestStmt instanceof DeclarationStatement)
        continue;

      //If Label statement move on
      if (closestStmt instanceof Label)
        continue;

      //If closest statement is part of the compute core don't bother - needed because compound stmt might be part
      //of core and not the statement itself
      boolean contained = false;
      Traversable t = (Traversable)closestStmt;
      while (t != null) {
        if (t instanceof Statement) {
          Statement tmpStmt = (Statement)t;
          if (mComputeStatements.contains(tmpStmt)) {
            contained = true;
            break;
          }
        }
        t = t.getParent();
      }
      if (contained)
        continue;

      // *FIXME* Now only looks for RAW dependencies where the COMPUTE does the Write
      //         We should also look for RAW where the COMPUTE does the Read
      currDefUseSet.clear();
      currDefUseSet.addAll(DataFlowTools.getUseSet(currExpr));

      //Check whether the def/use set of this stmt contains the variable being analyzed
      for (Expression useExpr : currDefUseSet) {
        if(useExpr instanceof IDExpression && !FCUDAutils.isCudaDim3Struct(useExpr)) {
          IDExpression idExpr = (IDExpression)useExpr;
          VariableDeclarator tmpDecl = FCUDAutils.getVariableDeclarator(idExpr);
          if (varDecl.equals(tmpDecl)) {

            // *TAN* if a variable is being used in a ForLoop, do not privatize it
            if (closestStmt instanceof ForLoop) {
              if (((ForLoop)closestStmt).getCondition().equals(currExpr) ||
                  ((ForLoop)closestStmt).getInitialStatement().equals(currExpr))
                return false;
            }
            System.out.println("Variable: "+varDecl.getSymbolName()+" used outside compute core: "
                +closestStmt.toString());
            returnVal = true;
            break;
          }
        }
      }

      if (returnVal)
        break;
    }
    currDefUseSet.clear();
    return returnVal;

  } // isDefUsedOutsideCompute()


  private void convertToArrays()
  {
    // Get thread-block dimensionality
    int numDims = 0;

    // Get ThreadBlock dimensions from pragma and declare as global constant variables
    int bdimX=1, bdimY=1, bdimZ=1;
    List<FcudaAnnotation> procAnnots = mProcedure.getAnnotations(FcudaAnnotation.class);

    if (procAnnots.isEmpty())
      throw new UnsupportedInput("FCUDA currently requires all functions to have a GRID pragma annotation");

    // Get block dimensions from GRID pragma
    for (FcudaAnnotation pAnnot : procAnnots) {
      if (((String)pAnnot.get("fcuda")).equals("grid")) {

        // X dimension
        if (pAnnot.get("x_dim") != null) {
          bdimX = Integer.parseInt((String) pAnnot.get("x_dim"));
          numDims = numDims + 1;
        }

        // Y dimension
        if (pAnnot.get("y_dim") != null) {
          bdimY = Integer.parseInt((String) pAnnot.get("y_dim"));
          numDims = numDims + 1;
        }

        // Z dimension
        if (pAnnot.get("z_dim") != null) {
          bdimZ = Integer.parseInt((String) pAnnot.get("z_dim"));
          numDims = numDims + 1;
        }
        FCUDAGlobalData.setKernTblkDim(mProcedure, numDims);
        break;
      }
    }

    IDExpression bdimXid = null, bdimYid = null, bdimZid = null;
    VariableDeclarator bdimXtor = null, bdimYtor = null, bdimZtor = null;
    LinkedList<Specifier> specLst = new LinkedList();
    specLst.add(Specifier.CONST);
    specLst.add(Specifier.INT);

    LinkedList<Declarator> declorLst = new LinkedList<Declarator>();
    bdimXid = (IDExpression) new NameID("BLOCKDIM_X" + "_" + mProcedure.getSymbolName());
    bdimXtor = new VariableDeclarator(bdimXid);
    bdimXtor.setInitializer(new Initializer((Expression) new IntegerLiteral(bdimX)));
    declorLst.add((Declarator)bdimXtor);

    if (numDims > 1) {
      bdimYid = (IDExpression) new NameID("BLOCKDIM_Y" + "_" + mProcedure.getSymbolName());
      bdimYtor = new VariableDeclarator(bdimYid);
      bdimYtor.setInitializer(new Initializer((Expression) new IntegerLiteral(bdimY)));
      declorLst.add((Declarator)bdimYtor);
    }

    if (numDims > 2) {
      bdimZid = (IDExpression) new NameID("BLOCKDIM_Z" + "_" + mProcedure.getSymbolName());
      bdimZtor = new VariableDeclarator(bdimZid);
      bdimZtor.setInitializer(new Initializer((Expression) new IntegerLiteral(bdimZ)));
      declorLst.add((Declarator)bdimZtor);
    }

    VariableDeclaration bdimDecl = new VariableDeclaration(specLst, declorLst);
    TranslationUnit thisfile=null;
    try {
      thisfile = (TranslationUnit) mProcedure.getParent();
    } catch (ClassCastException e) {
      Tools.exit("Isn't the parent of this procedure a translation unit?");
    }

    // Add declaration of block dimensions
    thisfile.addDeclarationBefore((Declaration) mProcedure, (Declaration) bdimDecl);


    // Create array specifier and convert to arrays
    LinkedList<Expression> arrayDims = new LinkedList<Expression>();
    if (numDims == 3)
      arrayDims.add(new Identifier(bdimZtor));
    if (numDims > 1)
      arrayDims.add(new Identifier(bdimYtor));
    arrayDims.add(new Identifier(bdimXtor));

    for (VariableDeclarator currDeclor : mVarsToConvert) {
      VariableDeclaration currDecl = FCUDAutils.getVariableDeclaration(currDeclor.getID());
      VariableDeclaration blockDecl = FCUDAutils.createSharedArrayDeclaration(currDecl, currDeclor, arrayDims);
      ((CompoundStatement)IRTools.getAncestorOfType(currDeclor, CompoundStatement.class)).addDeclarationAfter(currDecl,  blockDecl);

      // Remove scalar declaration
      Statement declStmt = (Statement)IRTools.getAncestorOfType(currDecl, Statement.class);
      declStmt.detach();

      // Replace scalar identifier with array identifier
      Expression blockAccess = FCUDAutils.getThreadIdxArrayAccess(blockDecl, numDims);
      Traversable t = (Traversable)mProcedure;
      IRTools.replaceAll(t, (Expression)currDeclor.getID(), blockAccess);
    }
    arrayDims.clear();
  } // convertToArrays()

  private void makeArraysInCompute(Procedure proc)
  {
    BreadthFirstIterator iter = new BreadthFirstIterator(proc);
    iter.pruneOn(Expression.class);
    FcudaAnnotation fcAnnot;
    String annotType;

    boolean storeStatements = false;
    while (iter.hasNext()) {
      Statement stmt = null;

      try {
        stmt = (Statement)iter.next(Statement.class);
        System.out.println("Statement: "+stmt.toString());
        System.out.flush();
      } catch (NoSuchElementException e) {
        break;
      }

      // Add statement into new procedure
      if (storeStatements && !(stmt instanceof AnnotationStatement)) {
        mComputeStatements.add(stmt);
        System.out.println("Inside compute: "+stmt.toString());
      }

      if (!(stmt instanceof AnnotationStatement))
        continue;

      // Only interested in COMPUTE FCUDA pragmas
      List<FcudaAnnotation> fcAnnots = stmt.getAnnotations(FcudaAnnotation.class);
      if (fcAnnots.isEmpty())
        continue;

      fcAnnot = fcAnnots.get(0);

      annotType = (String) fcAnnot.get("fcuda");

      if (!(annotType.equals("compute")))
        continue;

      if (((String)fcAnnot.get("begin")).equals("true"))
        storeStatements = true;

      if (fcAnnot.get("end")!=null && ((String)fcAnnot.get("end")).equals("true")) {
        //Find all vars that are defined in between compute pragmas that need to be converted to arrays
        findVarsToConvert();
        storeStatements = false;
        mComputeStatements.clear();
      }
    }
    convertToArrays();


  } // splitTasks()

  public void transformProcedure(Procedure proc)
  {
    if (!(FCUDAGlobalData.isConstMemProc(proc))) {
      mProcedure = proc;

      mComputeStatements = new HashSet<Statement>();
      mComputeStatements.clear();

      mDefUseSet = new HashSet<Expression>();
      mDefUseSet.clear();

      mVarsToConvert = new HashSet<VariableDeclarator>();
      mVarsToConvert.clear();

      makeArraysInCompute(proc);
    }
  }
}
