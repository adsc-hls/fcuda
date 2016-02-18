package fcuda.transforms;

import java.util.*;

import fcuda.utils.*;
import fcuda.ir.*;
import fcuda.analysis.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;

/**
 * Creates arrays for local data so that individual threads in a
 * serialized context can access a unique local store
 * Assumes that the single-declaration a serialize threads passes have
 * already run
 *
 * Future EnforceSyncs passes will likely make the optimizations
 * done in this pass unsound, so it is strongly encouraged that
 * this pass be run after all EnforceSyncs passes
 *
 * This pass cannot be run more than once per function
 *
 */
public class KernelStateTransform extends KernelTransformPass
{
  protected final LocalStateTransform lst;
  protected final BlockStateTransform bst;
  private GlobalKernelState stateStructures;

  public KernelStateTransform(Program program)
  {
    super(program);
    stateStructures = new GlobalKernelState();

    if (Driver.getOptionValue("structureLocalState") != null)
      lst = new StructureLocalState(stateStructures);
    else
      lst = new ArrayLocalState(stateStructures);
    if (Driver.getOptionValue("staticSharedState") != null)
      bst = new StaticSharedState(stateStructures);
    else
      bst = new PrivateSharedState();
  }

  public String getPassName()
  {
    return new String("[KernelStateTransform-MCUDA]");
  }


  public List<VariableDeclaration> getDeclsToPromote(Procedure proc,
      List<VariableDeclaration> candidates)
  {
    HashMap<CompoundStatement, List<Set<Symbol>>> blockAccesses =
      new HashMap<CompoundStatement, List<Set<Symbol>>>();
    for (CompoundStatement s : MCUDAUtils.getThreadLoops(proc.getBody())) {
      Set<Symbol> uses = Tools.getUseSymbol(s);
      Set<Symbol> defs = Tools.getDefSymbol(s);
      blockAccesses.put(s, (new ChainedList<Set<Symbol>>()).addLink(defs).addLink(uses));
    }

    // Tools doesn't have a pruned getUse function for Symbols, so we just build one
    Set<Symbol> outsideAccesses = new HashSet<Symbol>();
    System.out.println(">>> outside uses:\n"+DataFlowTools.getUseSet(proc.getBody(),ThreadLoop.class));
    for (Expression e : DataFlowTools.getUseSet(proc.getBody(),ThreadLoop.class)) {
      Expression tmpExp = e;
      if (e instanceof AccessExpression) {
        tmpExp = ((AccessExpression)e).getLHS();
        while (SymbolTools.getSymbolOf(tmpExp) instanceof AccessSymbol) {
          if (tmpExp instanceof ArrayAccess)
            tmpExp = ((ArrayAccess)tmpExp).getArrayName();
          if (tmpExp instanceof AccessExpression)
            tmpExp = ((AccessExpression)tmpExp).getLHS();
        }
      }
      Symbol s = SymbolTools.getSymbolOf(tmpExp);
      if (s != null) {
        System.out.println(">>> handling: "+((VariableDeclarator)s).toString());
        outsideAccesses.add(s);
      }
    }
    //This function is a hack around emulating
    //a live-range analysis.  We're trying to see if the live
    //range of a variable fits within a single block.
    //
    //Eventually, this should be coupled with some kind of SSA,
    //so that variables that have live ranges in different blocks,
    //but none that cross block boundaries, can also be detected.

    List<VariableDeclaration> vDeclstoPromote = new LinkedList<VariableDeclaration>();
    for (VariableDeclaration vDecl : candidates) {

      //First check.  If the declaration for the variable is
      //contained within an existing thread loop, it doesn't
      //need to be buffered.
      if (MCUDAUtils.getThreadLoopBody(vDecl) != null)
        continue;

      //Second Check: Count the number of blocks in which this variable is
      //accessed.  If the answer is 1 or 0 (unused variable?), don't buffer.
      int numBlocksDef = 0, numBlocksRef = 0;

      for (CompoundStatement s : MCUDAUtils.getThreadLoops(vDecl.getParent().getParent())) {
        List<Set<Symbol>> defuse = blockAccesses.get(s);
        Set<Symbol> blockDefs = defuse.get(0);
        Set<Symbol> blockUses = defuse.get(1);

        if (blockDefs.contains(vDecl.getDeclarator(0)))
          numBlocksDef++;
        else if (blockUses.contains(vDecl.getDeclarator(0)))
          numBlocksRef++;
      }

      if (numBlocksDef > 1 || (numBlocksDef != 0 && numBlocksRef > 0) ||
          numBlocksDef > 0 && outsideAccesses.contains(vDecl.getDeclaredIDs().get(0))) {
        System.out.println("adding decl to promote");
        vDeclstoPromote.add(vDecl);
          }
    }

    return vDeclstoPromote;
  } // getDeclsToPromote()


  public void transformProcedure(Procedure proc)
  {
    DataDepAnalysis datadep = new DataDepAnalysis(proc);

    String procName = proc.getName().toString();
    stateStructures.lvStructID = new NameID(procName + "_local_state");
    IDExpression kernelStructID = new NameID(procName + "_block_state");
    stateStructures.SM_state = new ClassDeclaration(ClassDeclaration.STRUCT, kernelStructID);
    stateStructures.SM_data_var = new NameID("tb_data");

    stateStructures.threadState = new ClassDeclaration(ClassDeclaration.STRUCT,
        stateStructures.lvStructID);
    stateStructures.kernelVars = new NameID(procName + "_vars");
    stateStructures.localVars = new NameID("lv");

    DepthFirstIterator iter = new DepthFirstIterator(proc.getBody());
    List<VariableDeclaration> vDeclList = iter.getList(VariableDeclaration.class);
    List<VariableDeclaration> sharedVDecls = new LinkedList<VariableDeclaration>();
    List<VariableDeclaration> temp = new ChainedList<VariableDeclaration>().addAllLinks(vDeclList);

    for (VariableDeclaration vDecl : temp) {
      boolean isLocalVar = true;
      IDExpression varname = vDecl.getDeclaredIDs().get(0);
      //Introduced thread index variable doesn't count
      if (varname.compareTo(MCUDAUtils.LocalTidx.getId().get(0)) == 0)
        isLocalVar = false;

      //Check if it's a shared variable to add to that list
      if (vDecl.getSpecifiers().contains(Specifier.SHARED)) {
        //Add it to the shared list, unless it's a standin Tidx
        if (varname.compareTo(MCUDAUtils.Tidx.getId().get(0)) != 0)
          sharedVDecls.add(vDecl);
        isLocalVar = false;
      }

      if (!isLocalVar || !datadep.isThrDepVar(varname) )
        vDeclList.remove(vDecl);
    }

    List<VariableDeclaration> vDeclstoPromote;
    vDeclstoPromote = getDeclsToPromote(proc, vDeclList);

    System.out.println("transforming Decls");
    lst.TransformLocals(vDeclstoPromote, proc);
    bst.TransformSharedVars(sharedVDecls);

    if (!stateStructures.SM_state.getDeclarations().isEmpty()) {
      TranslationUnit tu = (TranslationUnit)proc.getParent();
      tu.addDeclarationBefore(proc, stateStructures.SM_state);
      MCUDAUtils.addStructTypedefBefore(tu, proc, stateStructures.SM_state);

      VariableDeclarator sm_declarator = new VariableDeclarator(stateStructures.kernelVars,
          new ArraySpecifier(
            new IntegerLiteral(
              Integer.parseInt(
                Driver.getOptionValue("numRuntimeBlocks")))));
      VariableDeclaration kernelState = new VariableDeclaration(
          new UserSpecifier(
            (IDExpression)kernelStructID.clone()), sm_declarator );
      tu.addDeclarationBefore(proc, kernelState);

      //Add a variable declaration for terser access to the structure
      ArrayAccess SM_struct = new ArrayAccess((Expression)stateStructures.kernelVars.clone(), MCUDAUtils.getSMID());
      VariableDeclarator SM_data_ref = new VariableDeclarator(
          PointerSpecifier.UNQUALIFIED,
          (IDExpression)stateStructures.SM_data_var.clone());

      SM_data_ref.setInitializer(new Initializer(
            new UnaryExpression(
              UnaryOperator.ADDRESS_OF, SM_struct)));
      VariableDeclaration SM_data_decl = new VariableDeclaration(
          new UserSpecifier(
            (IDExpression)kernelStructID.clone()), SM_data_ref );
      if (proc.getBody().getChildren().get(0) instanceof DeclarationStatement)
        proc.getBody().addStatementBefore(
            (Statement)proc.getBody().getChildren().get(0),
            new DeclarationStatement(SM_data_decl));
      else
        proc.getBody().addDeclaration(SM_data_decl);
    }
  } // transformProcedure()
}
