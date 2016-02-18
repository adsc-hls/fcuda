package cetus.analysis;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import cetus.analysis.CallGraph.Node;
import cetus.hir.AccessExpression;
import cetus.hir.Annotatable;
import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.ClassDeclaration;
import cetus.hir.CommaExpression;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FlatIterator;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.GotoStatement;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.InlineAnnotation;
import cetus.hir.IntegerLiteral;
import cetus.hir.Label;
import cetus.hir.NameID;
import cetus.hir.NestedDeclarator;
import cetus.hir.NullStatement;
import cetus.hir.PointerSpecifier;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.ReturnStatement;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTable;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.UserSpecifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.transforms.SingleReturn;

/**
 * Performs simple function inlining. It assumes that the code to be inlined is already compiled correctly and has no errors.
 * Following are taken care of: 
 * 		* Functions that result in recursion are not inlined
 * 		* Functions that use static external variables are not inlined
 * 		* Global variables used in the function to be inlined are handled by declaring them as extern
 * 		* Variable reshaping (array-dimensions) is handled
 * 		* Long function call chains (a->b->c->d->...) are handled as long as they do not result in a recursion
 * 		* Switches are provided for controlling variable names and their length
 * 		* Support for making log
 * 		* Comments with the inlined code
 */
public class InlineExpansion {
	
	/** Default option for making log */
	private static final boolean MAKE_LOG = true;
	/** Default prefix for the new variable name for a parameter */
	private static final String PARAM_PREFIX = "param";
	/** Default prefix for the new variable name for a local variable */
	private static final String LOCAL_PREFIX = "local";
	/** Default prefix for the new name for a label */
	private static final String LABEL_PREFIX = "label";
	/** Default prefix for the name of the variable used to hold the return value of the function that has been inlined */
	private static final String RESULT_PREFIX = "result";
	/** Default maximum length of variables' names that inlining introduces */
	private static final int MAX_VAR_LENGTH = 256;
	/** Default option for variable naming, in case of fully qualified names name of function is also appended, short form does not include that */
	private static final boolean FULLY_QUALIFIED_NAMES = true;
	/** Default option for functions that have local static variables, we won't inline these functions, by default */
	private static final boolean INLINE_FUNCS_WITH_STATIC_VARS = false;
	
	/** Default mode for inlining, inlining will be performed inside main function */
	public static final int MODE_DEFAULT = 0;
	/** Inlining will be performed inside the selected functions provided on command line via "functions" option */
	public static final int MODE_INSIDE_SELECTED_FUNCTIONS = 1;
	/** The selected functions provided on command line will be inlined wherever they are invoked */
	public static final int MODE_SELECTED_FUNCTIONS_INVOCATIONS = 2;
	/** Starting from main function inline the code which is inside "inlinein" pragmas */
	public static final int MODE_INLINEIN_PRAGMAS = 3;
	/** Same as "MODE_INLINEIN_PRAGMAS" except that "inline" pragmas will also be honored */
	public static final int MODE_INLINE_PRAGMAS = 4;
	
	/** By default "noinlinein" and "noinline" pragmas will not be honored */
	private static final boolean HONOR_NO_PRAGMAS = true;
	/** By default debug option is turned off, which means that the inlined functions will be removed if they are no longer being executed */
	private static final boolean DEBUG = false;
	/** By default, the functions given on the command line would be taken as such and not their complement (all others except these) */
	private static final boolean COMPLEMENT_FUNCTIONS = false;
	/** By default, inlining would be performed recursively, i.e. within the callees and their callees and so on. */
	private static final boolean LEVEL_1 = false;
	/** By default, all function calls will be inlined and not just those which are inside for loops */
	private static final boolean INSIDE_FOR_LOOPS_ONLY = false;
	/** Current option for making log */
	private boolean makeLog = MAKE_LOG;
	/** Currently used prefix for the new variable name for a parameter */
	private String paramPrefix = PARAM_PREFIX;
	/** Currently used prefix for the new variable name for a local variable */
	private String localVarPrefix = LOCAL_PREFIX;
	/** Currently used prefix for the new name for a label */
	private String labelPrefix = LABEL_PREFIX;
	/** Currently used prefix for the name of the variable used to hold the return value of the function that has been inlined */
	private String resultPrefix = RESULT_PREFIX;
	/** Currently used maximum length of variables' names that inlining introduces */
	private int maxVarLength = MAX_VAR_LENGTH;
	/** Current option for variable naming, in case of fully qualified names name of function is also appended, short form does not include that */
	private boolean fullyQualifiedName = FULLY_QUALIFIED_NAMES;
	/** Current option for inlining functions that have static local variables */
	private boolean inlineFuncsWithStaticVars = INLINE_FUNCS_WITH_STATIC_VARS;
	/** Current mode of the inliner */
	private int mode = MODE_DEFAULT;
	/** Current boolean value of honor "nolinein" and "noinline" pragmas option */
	private boolean honorNoPragmas = HONOR_NO_PRAGMAS;
	/** Current boolean value of debug option, if this option is on we will not delete the unreachable functions after inlining */
	private boolean debug = DEBUG;
	/** Current boolean value of level-1 option, if this option is on inlining will be restrained to level-1 (and will not be recursive) */
	private boolean level_1 = LEVEL_1;
	/** Current boolean value of foronly option, if this option is on, only the function calls inside for loops will be inlined */
	private boolean insideForOnly = INSIDE_FOR_LOOPS_ONLY;
	/** 
	 * Current boolean value of complement functions option, if this option is on all functions except the ones provided on the command
	 *  line will be inlined according to the selected mode 
	 */
	private boolean complementFunctions = COMPLEMENT_FUNCTIONS;
	/** Function names provided on the command line (will be used by MODE_INSIDE_SELECTED_FUNCTIONS and MODE_SELECTED_FUNCTIONS_INVOCATIONS modes) */
	private ArrayList<String> functionNames = new ArrayList<String>();
	/** 
	 * List containing functions that are to be considered for MODE_INSIDE_SELECTED_FUNCTIONS and MODE_SELECTED_FUNCTIONS_INVOCATIONS modes 
	 *  Note that this list is emptied after every run of inline(Program) 
	 */
	private ArrayList<Procedure> functions = new ArrayList<Procedure>();
	/** The functions that were inlined as a result to the inline(Program) call. This list would be used while handling debug option */
	private ArrayList<Procedure> inlinedFunctions = new ArrayList<Procedure>();
	
	/**
	 * default constructor
	 */
	public InlineExpansion() {
		mode = MODE_DEFAULT;
	}
	/**
	 * @param mode - the mode for the inliner, should be one of the MODE_XXX constants
	 */
	public InlineExpansion(int mode) {
		this();
		setMode(mode);
	}
	/**
	 * Creates an instance of the Inliner
	 * @param mode - mode for the inliner, should be one of the MODE_XXX constants
	 * @param functions - ArrayList containing function names provided by the user on the command line
	 *                  - it will be used by MODE_INSIDE_SELECTED_FUNCTIONS and MODE_SELECTED_FUNCTIONS_INVOCATIONS modes 
	 */
	public InlineExpansion(int mode, ArrayList<String> functions) {
		this(mode);
		this.functionNames.addAll(functions);
	}
	/**
	 * sets the make-log flag
	 * @param makeLog - new value for the currently used make-log option
	 */
	public void setMakeLog(boolean makeLog) {
		this.makeLog = makeLog;
	}
	/**
	 * sets the currently used prefix for new variables replacing old function parameters
	 * @param paramPrefix - the new prefix
	 */
	public void setParamPrefix(String paramPrefix) {
		this.paramPrefix = paramPrefix;
	}
	/**
	 * sets the currently used prefix for new variables used to hold return value of the function that has been inlined
	 * @param resultPrefix - the new prefix
	 */
	public void setResultPrefix(String resultPrefix) {
		this.resultPrefix = resultPrefix;
	}
	/**
	 * sets the currently used prefix for new variables replacing old local variables
	 * @param localVarPrefix - the new prefix for local variables in the inlined code
	 */
	public void setLocalVarPrefix(String localVarPrefix) {
		this.localVarPrefix = localVarPrefix;
	}
	/**
	 * sets the currently used prefix for new labels replacing old labels
	 * @param labelPrefix - the new prefix for labels in the inlined code
	 */
	public void setLabelPrefix(String labelPrefix) {
		this.labelPrefix = labelPrefix;
	}
	/**
	 * sets the maximum length for the variables introduced by inlining
	 * @param length - new length for variables
	 */
	public void setMaxVarLength(int length) {
		this.maxVarLength = length;
	}
	/**
	 * sets the option to have verbose variable names
	 * @param verboseName - the option to have verbose names
	 */
	public void setFullyQualifiedName(boolean verboseName) {
		this.fullyQualifiedName = verboseName;
	}
	/**
	 * sets the option to inline functions that have static local variables
	 * @param inline - the option to inline such function
	 */
	public void setInlineFuncsWithStaticVars(boolean inline) {
		this.inlineFuncsWithStaticVars = inline;
	}
	/**
	 * sets the mode for the inliner
	 * @param mode - the mode value, should be one of the MODE_XXX constants
	 */
	public void setMode(int mode) {
		if(mode == MODE_DEFAULT || mode == MODE_INSIDE_SELECTED_FUNCTIONS || mode == MODE_SELECTED_FUNCTIONS_INVOCATIONS
				|| mode == MODE_INLINEIN_PRAGMAS || mode == MODE_INLINE_PRAGMAS) {
			this.mode = mode;
		}	
	}
	/**
	 * sets the function names given on command line for MODE_INSIDE_SELECTED_FUNCTIONS and MODE_SELECTED_FUNCTIONS_INVOCATIONS modes 
	 * @param functions - function names provided on the command line
	 */
	public void setCommandlineFunctions(ArrayList<String> functions) {
		this.functionNames.clear();
		this.functionNames.addAll(functions);
	}
	/**
	 * sets the debug option, which when on, will not result in deleting the unreachable functions after inlining
	 * @param debug - the boolean value for debug option
	 */
	public void setDebugOption(boolean debug) {
		this.debug = debug;
	}
	/**
	 * sets the honor pragmas option
	 * @param honorPragmas - the boolean value for honor pragmas option
	 */
	public void setHonorPragmas(boolean honorPragmas) {
		this.honorNoPragmas = honorPragmas;
	}
	/**
	 * sets the level_1 option, which when on, will result in 1-level inlining as opposed to recursive 
	 * @param level_1 - the boolean value for level_1 option
	 */
	public void setLevel_1(boolean level_1) {
		this.level_1 = level_1;
	}
	/**
	 * sets the inside foronly option, which when on, will instruct the inliner to try to inline the function calls inside for loops only 
	 * @param insdeForOnly - the boolean value for foronly option
	 */
	public void setInsideForOnly(boolean insideForOnly) {
		this.insideForOnly = insideForOnly;
	}
	/**
	 * sets the complement functions option, which when on, will consider all functions for inlining according to the provided mode,
	 * except the ones given in the command line
	 * @param complementFunctions - the boolean value for complement functions option
	 */
	public void setComplementFunctions(boolean complementFunctions) {
		this.complementFunctions = complementFunctions;
	}
	
	/**
	 * tells if the log is being made
	 */
	public boolean isMakeLog() {
		return makeLog;
	}
	/**
	 * returns the prefix used in naming variables that replace parameters in the function to be inlined
	 */
	public String getParamPrefix() {
		return paramPrefix;
	}
	/**
	 * returns the prefix used in naming variables that replace local variables in the inlined code
	 */
	public String getLocalVarPrefix() {
		return localVarPrefix;
	}
	/**
	 * returns the prefix used in naming labels that replace labels in the inlined code
	 */
	public String getLabelPrefix() {
		return labelPrefix;
	}
	/**
	 * returns the prefix used in naming the variable that holds the return value of the function that has been inlined
	 */
	public String getResultPrefix() {
		return resultPrefix;
	}
	/**
	 * returns the maximum allowed length for the variable names that inlining introduces
	 */
	public int getMaxVarLength() {
		return maxVarLength;
	}
	/**
	 * tells if the inlining code uses verbose names for variables that it introuduces
	 */
	public boolean isFullyQualifiedName() {
		return fullyQualifiedName;
	}
	/**
	 * tells if the functions having static local variables will be inlined or not
	 */
	public boolean isInlineFuncsWithStaticVars() {
		return inlineFuncsWithStaticVars;
	}
	/**
	 * tells the mode of the inliner, which is one of the MODE_XXX constants
	 */
	public int getMode() {
		return mode;
	}
	/**
	 * returns an array list of function names provided at the command line for MODE_INSIDE_SELECTED_FUNCTIONS and MODE_SELECTED_FUNCTIONS_INVOCATIONS modes
	 * @return
	 */
	public ArrayList<String> getCommandlineFunctions() {
		return this.functionNames;
	}
	/**
	 * tells if the debug option is on
	 */
	public boolean isDebugOption() {
		return this.debug;
	}
	/**
	 * tells if the honor pragmas option is on
	 */
	public boolean isHonorPragmas() {
		return this.honorNoPragmas;
	}
	/**
	 * tells if the level_1 option is on 
	 */
	public boolean isLevel_1() {
		return this.level_1;
	}
	/**
	 * tells if the foronly option is on 
	 */
	public boolean isInsideForOnly() {
		return this.insideForOnly;
	}
	/**
	 * tells if the complement functions option is on
	 */
	public boolean isComplementFunctions() {
		return this.complementFunctions;
	}
	
	/**
	 * performs inlining in the given program according to the options that have been set before calling this function
	 * Following are taken care of:
	 * 		* Functions that result in recursion are not inlined
	 * 		* Functions that use static external variables are not inlined
	 * 		* Global variables used in the function to be inlined are handled by declaring them as extern
	 * 		* Variable reshaping (array-dimensions) is handled
	 * 		* Long function call chains (a->b->c->d->...) are handled as long as they do not result in a recursion
	 * 		* Switches are provided for controlling variable names and their lengths
	 * 		* Support for making log
	 * 		* Comments with the inlined code
	 *   
	 * @param program - the program
	 */
	public void inline(Program program) {
		
		// make the call graph
	    CallGraph callGraph = new CallGraph(program);
	    ArrayList<Procedure> notInlineIn = new ArrayList<Procedure>();
	    if(functionNames.size() > 0 && (mode == MODE_INSIDE_SELECTED_FUNCTIONS || mode == MODE_SELECTED_FUNCTIONS_INVOCATIONS)) {
		    List<Procedure> procs = callGraph.getTopologicalCallList();
		    for (Procedure procedure : procs) {
		    	String name = procedure.getSymbolName();
		    	if(functionNames.contains(name)) {
		    		if(!complementFunctions) {
		    			functions.add(procedure);
		    		}
		    		else {
		    			notInlineIn.add(procedure);
		    		}
		    	}
		    	else {
		    		if(complementFunctions) {
		    			functions.add(procedure);
		    		}	
		    	}
			}
	    }
	    inlinedFunctions.clear();
	    if(mode == MODE_INSIDE_SELECTED_FUNCTIONS) {
	    	if(!level_1) {
	    		functions = getFunctionsRecursively(callGraph, functions, notInlineIn);
	    	}
    		for (Procedure procedure : functions) {
    			internalInlineIn(procedure.getBody(), procedure, callGraph);
    		}
	    }
	    else {
		    Procedure main = null;
		    // first inline the callee functions deep in the call graph, do it bottom up
		    List<Procedure> procs = callGraph.getTopologicalCallList();
		    for (Procedure procedure : procs) {
		    	if(procedure.getSymbolName().equals("main") || procedure.getSymbolName().equals("Main__")) {
		    		main = procedure;
		    	}
		    	else if(!(level_1)) {
		    		internalInlineIn(procedure.getBody(), procedure, callGraph);
		    	}	
			}
		    
		    // now inline the main procedure
		    if(main != null) {
		    	internalInlineIn(main.getBody(), main, callGraph);
		    }	
	    }
	    // empty the functions list, it is only used for one run
	    functions.clear();

	    // handle the debug switch if it was on
	    if(!debug) {
	    	callGraph = new CallGraph(program);
	    	List<Procedure> procs = callGraph.getTopologicalCallList();
	    	List<Procedure> allProcs = new ArrayList<Procedure>();
	    	for(Traversable t : program.getChildren()) {
	    		if(t instanceof TranslationUnit) {
	    			for (Traversable t2 :((TranslationUnit)t).getChildren()) {
	    	    		if(t2 instanceof Procedure) {
	    	    			allProcs.add((Procedure)t2);
	    	    		}	
	    			}
	    		}
	    	}
//		    for (Procedure p : inlinedFunctions) {
		    for (Procedure p : allProcs) {
		    	if(!procs.contains(p)) {
		    		p.getParent().removeChild(p);
		    	}
		    }	
	    }
	}
	
	/**
	 * inlines the function calls in the body of the given procedure, doesn't go deeper than that
	 * i.e. does not inline inside functions called by the procedure recursively.
	 * @param proc - the procedure 
	 *                  
	 * Following are taken care of:
	 * 		* Function calls that result in recursion are not inlined
	 * 		* Functions that use static external variables are not inlined
	 * 		* Global variables used in the function to be inlined are handled by declaring them as extern
	 * 		* Variable reshaping (array-dimensions) is handled
	 * 		* Switches are provided for controlling variable names and their lengths
	 * 		* Support for making log
	 * 		* Comments with the inlined code
	 */
	public void inlineIn(Procedure proc) {		
		internalInlineIn(proc.getBody(), proc, new CallGraph(getProgram(proc)));
	}	
	
	/**
	 * inlines the function calls in the given compound statement, doesn't go deeper than that
	 * i.e. does not inline inside functions called from the given compound statement.
	 * @param inlineIn - the compound statement in which inlining is to be performed 
	 *                  
	 * Following are taken care of:
	 * 		* Function calls that result in recursion are not inlined
	 * 		* Functions that use static external variables are not inlined
	 * 		* Global variables used in the function to be inlined are handled by declaring them as extern
	 * 		* Variable reshaping (array-dimensions) is handled
	 * 		* Switches are provided for controlling variable names and their lengths
	 * 		* Support for making log
	 * 		* Comments with the inlined code
	 */
	public void inlineIn(CompoundStatement inlineIn) {
		Procedure proc = getEnclosingFunction(inlineIn);
		internalInlineIn(inlineIn, proc, new CallGraph(getProgram(proc)));		
	}
	
	/**
	 * inlines the function calls in the given compound statement, doesn't go deeper than that
	 * i.e. does not inline inside functions called by the procedure recursively.
	 * @param inlineIn - the compound statement in which inlining is to be performed
	 * @param enclosingFunction - the function that encloses the given compound statement inside which inling is to be performed 
	 * @param callGraph - the call graph, using call graph this method makes sure that the function calls that result in recursion
	 *                  - are not inlined
	 *                  
	 * Following are taken care of:
	 * 		* Function calls that result in recursion are not inlined (it can be bypassed by passing null for the call graph)
	 * 		* Functions that use static external variables are not inlined
	 * 		* Global variables used in the function to be inlined are handled by declaring them as extern
	 * 		* Variable reshaping (array-dimensions) is handled
	 * 		* Switches are provided for controlling variable names and their lengths
	 * 		* Support for making log
	 * 		* Comments with the inlined code
	 */
	private void internalInlineIn(CompoundStatement inlineIn, Procedure enclosingFunction, CallGraph callGraph) {
		String enclosingFunctionName = enclosingFunction.getName().toString();
	    CompoundStatement enclosingFunctionBody = enclosingFunction.getBody();
	    InlineAnnotation procInlineInAnnotation = getInlineInAnnotation(enclosingFunction.getAnnotations(InlineAnnotation.class));
	    
	    // if we are honoring "noinlinein" pragma and the procedure has "noinlinein" pragma we can't perform inlining in it
	    if(honorNoPragmas && InlineAnnotation.NO_INLINE_IN.equals(procInlineInAnnotation)) {
	    	return;
	    }
	    
	    DepthFirstIterator dfi = new DepthFirstIterator(inlineIn);
	    Vector<Statement> statementsWithFunctionCalls = new Vector<Statement>();
	    
	    // gather all the statements that have function calls
	    Object obj;
	    while(dfi.hasNext()) {
	    	if((obj = dfi.next()) instanceof FunctionCall){
		    	FunctionCall call = (FunctionCall)obj;
	    		Statement s = getStatement(call);
	    		if(!statementsWithFunctionCalls.contains(s))
	    			statementsWithFunctionCalls.addElement(s);
	    	}	
	    }
	    	
	    // for each statement, inline every function that is called from it 
	STATEMENT_LOOP:    for(int i = 0; i < statementsWithFunctionCalls.size(); i++){
	    	
	    	Statement statementWithFunctionCall = statementsWithFunctionCalls.elementAt(i);
	    	CompoundStatement enclosingCompoundStmt = getEnclosingCompoundStmt(statementWithFunctionCall);
	    	if(enclosingCompoundStmt == null)
	    		enclosingCompoundStmt = enclosingFunctionBody;
	    
	    	if(mode == MODE_INLINEIN_PRAGMAS || mode == MODE_INLINE_PRAGMAS) {
	    		// if the function is not preceded by inline pragma we got to inline only those statement which are preceded by inlinein 
	    		if(!InlineAnnotation.INLINE_IN.equals(procInlineInAnnotation)) {
	    			
	    			Statement stmt = statementWithFunctionCall;
	    			boolean canInline = false;
	    			while(stmt != null) {
	    				InlineAnnotation inlineInAnnotation = getInlineInAnnotation(stmt.getAnnotations(InlineAnnotation.class));
	    				if(InlineAnnotation.INLINE_IN.equals(inlineInAnnotation)) {	    					
	    					canInline = true;
	    					break;
	    				}
	    				if(stmt.getParent() instanceof Procedure) {
	    					break;
	    				}
	    				stmt = (Statement)stmt.getParent();
	    			}
	    			if(!canInline) {
	    				continue STATEMENT_LOOP;
	    			}
	    		}
	    	}
	    	if(insideForOnly) {
    			Statement stmt = statementWithFunctionCall;
    			boolean canInline = false;
    			while(stmt != null) {
    				if(stmt instanceof ForLoop) {	    					
    					canInline = true;
    					break;
    				}
    				if(stmt.getParent() instanceof Procedure) {
    					break;
    				}
    				stmt = (Statement)stmt.getParent();
    			}
    			if(!canInline) {
    				continue STATEMENT_LOOP;
    			}
	    	}
	    	// if we are honoring "noinlinein" pragmas then we got to check if the statement or its parents are preceded by "noinlinein" pragma
	    	// in which case we won't inline
	    	if(honorNoPragmas) {
    			Statement stmt = statementWithFunctionCall;
    			while(stmt != null) {
    				InlineAnnotation noInlineIn = getInlineInAnnotation(stmt.getAnnotations(InlineAnnotation.class));
    				if(InlineAnnotation.NO_INLINE_IN.equals(noInlineIn)) {
    					continue STATEMENT_LOOP;
    				}
    				if(stmt.getParent() instanceof Procedure) {
    					break;
    				}
    				stmt = (Statement)stmt.getParent();
    			}
	    	}
	    	
	    	Statement firstNonDeclarationStmt = getFirstNonDeclarationStatement(enclosingCompoundStmt);
		    if(firstNonDeclarationStmt == null && statementsWithFunctionCalls.size() > 0){
	    		firstNonDeclarationStmt = statementsWithFunctionCalls.elementAt(0);
		    }
	    	
	    	if(Tools.indexByReference(enclosingCompoundStmt.getChildren(), statementWithFunctionCall) < 
	    	      Tools.indexByReference(enclosingCompoundStmt.getChildren(), firstNonDeclarationStmt)){

	    		if(makeLog){
					System.out.println("function calls in following statement in fucntion \'" + enclosingFunctionName + "\' can't be inlined because this statement appears before declaration statement(s) or is one itself");
					System.out.println(statementWithFunctionCall.toString());
	    		}	
				continue;
	    	}
	    	
	    	// There may be many function calls inside one statement
	    	dfi = new DepthFirstIterator(statementWithFunctionCall);
	    	Vector<FunctionCall> callsInStatement = new Vector<FunctionCall>();
	    	while(dfi.hasNext()){
    			if( (obj = dfi.next()) instanceof FunctionCall){
	    			callsInStatement.addElement((FunctionCall) obj);
    			}
	    	}
	    	
	    	// for each function call, inline the called function. Make sure to do it in reverse order.
	    	for (int j = callsInStatement.size()-1; j >= 0; j--) {
				
    			FunctionCall fc = callsInStatement.elementAt(j);
//    			Procedure function = (Procedure)(fc.getProcedure().clone()); // What? Procedure can't be cloned.. why? Okay, clone the body then. (note: in latest code it can be cloned)
    			Procedure function2BInlined = fc.getProcedure();
    			boolean inline = true;

    			// function can be null, e.g. in case if the function call involved function pointer(s)
    			if(function2BInlined == null){
    				if(makeLog){
    					System.out.println("function call in following statement in function \'" + enclosingFunctionName + "\' calls unrecognized or library function, or possibly involves function pointer(s) so we are not inlining it");
    					System.out.println(getStatement(fc).toString());
    				}	
    				continue;
    			}
    			// if we are only inling selected functions whenever invoked, and this function is not one of them, we got to skip it
    			if(mode == MODE_SELECTED_FUNCTIONS_INVOCATIONS && !functions.contains(function2BInlined)) {
    				continue;
    			}
    			// if we are honoring "noinline" and "noinlinein" pragmas and this function has "noinline" pragma, we cannot inline it
    			if(honorNoPragmas) {
	    			InlineAnnotation functionAnnotation = getInlineAnnotation(function2BInlined.getAnnotations(InlineAnnotation.class));
	    			if(InlineAnnotation.NO_INLINE.equals(functionAnnotation)) {
	    				continue;
	    			}
    			}
    			// if we are honoring "inline" function pragma and this function does not have "inline" pragma, we cannot inline it
    			if(mode == MODE_INLINE_PRAGMAS) {
	    			InlineAnnotation functionAnnotation = getInlineAnnotation(function2BInlined.getAnnotations(InlineAnnotation.class));
	    			if(!(InlineAnnotation.INLINE.equals(functionAnnotation))) {
	    				continue;
	    			}
    			}
    			
    			String functionName2BInlined = function2BInlined.getName().toString();
    			
    			// don't expand if it is a recursive call (self or otherwise).
    			if(callGraph != null && callGraph.isRecursive(function2BInlined)){
    				if(makeLog){
    					System.out.println("calling " + functionName2BInlined + " function in the following statement in function \'" + enclosingFunctionName + "\' results in a recursion around the function so we are not inlining it");
    					System.out.println(getStatement(fc).toString());
    				}
    				continue;
    			}
    			// self-recursion
    			if(function2BInlined.equals(enclosingFunction)){
    				if(makeLog){
    					System.out.println("calling " + functionName2BInlined + " function in the following statement in function \'" + enclosingFunctionName + "\' results in self-recursion so we are not inlining it");
    					System.out.println(getStatement(fc).toString());
    				}	
    				continue;
    			}
    			// check other conditions (like use of function pointers or static variables)
    			if(!canInline(function2BInlined, fc, enclosingFunction)){
    				continue;
    			}
    			
				// get the arguments and return type of the function that is to be inlined
    			List<Expression> args = fc.getArguments();
    			List<Specifier> returnTypes = function2BInlined.getReturnType();
    			List parameters = function2BInlined.getParameters();

				List<String> newParamNames = new LinkedList<String>();
    			List<IDExpression> oldParams = new LinkedList<IDExpression>();
    			List<Symbol> newSymbols = new LinkedList<Symbol>();
    			
    			Vector<Declaration> declarations = new Vector<Declaration>();
    			Vector<Statement> statements = new Vector<Statement>();
    			Vector<Declaration> addedGlobalDeclarations = new Vector<Declaration>();
    			Vector<Declaration> removedGlobalDeclarations = new Vector<Declaration>();
    			
    			
				// get the code to be inlined (don't forget to clone)
				CompoundStatement codeToBeInlined = (CompoundStatement)(function2BInlined.getBody().clone());

				// if there are multiple return statements in the function, use the single return transformation to have only one
				if(hasMultipleReturnStmts(codeToBeInlined)){
					Procedure p = (Procedure)function2BInlined.clone();
					new SingleReturn(getProgram(function2BInlined)).transformProcedure(p);
					function2BInlined = p;
					codeToBeInlined = p.getBody();
					codeToBeInlined.setParent(null);
				}
				
				// rename local variables in the inlined code, except for extern declarations
		        dfi = new DepthFirstIterator(codeToBeInlined);
		        List<IDExpression> locals = new LinkedList<IDExpression>();
		        List <IDExpression> labels = new LinkedList<IDExpression>();
		        List<String> newLocals = new LinkedList<String>();
		        // get all the local variables
		        obj = null;
		        while (dfi.hasNext()){
		        	if( (obj = dfi.next()) instanceof DeclarationStatement){
		        		Declaration d = ((DeclarationStatement)obj).getDeclaration();
		        		// skip the extern declarations
		        		if(d instanceof VariableDeclaration){
		        			List specs = ((VariableDeclaration)d).getSpecifiers();
		        			if(specs.contains(Specifier.EXTERN))
		        				continue;

		        			List<IDExpression> ids = ((VariableDeclaration)d).getDeclaredIDs();
		    				for (int k = 0; k < ids.size(); k++) {
								locals.add((IDExpression)(ids.get(k).clone())); // don't forget to clone
							}
		        		}
		        	}
		        	// also handle labels
		        	else if(obj instanceof Label){
		        		labels.add(((Label)obj).getName());
		        	}
				}
		        // come up with unique names for new variables and labels.
		        for (int k = 0; k < locals.size(); k++) {
					newLocals.add(getUniqueIdentifier(enclosingCompoundStmt, localVarPrefix + "_" + functionName2BInlined, locals.get(k).toString()).getName());
				}
		        for (int k = 0; k < labels.size(); k++) {
		        	locals.add(labels.get(k));
					newLocals.add(getUniqueLabel(enclosingFunction, labelPrefix + "_" + functionName2BInlined, labels.get(k).toString()).getName());
				}
		        HashMap<String, ArrayAccess> actualArgs = new HashMap<String, ArrayAccess>();
		        HashMap<IDExpression, Expression> offsets = new HashMap<IDExpression, Expression>();
		        // rename local variables
		        renameLocals(codeToBeInlined, locals, newLocals);
				// handle labels and name changing in annotations
		        replaceVariableNames(codeToBeInlined, locals, newLocals, new HashMap<String, ArrayAccess>(), new HashMap<IDExpression, Expression>(), new LinkedList<Symbol>(), false);
		        
    			if(parameters.size() == args.size()){
			        
    				// for each parameter, come up with a new variable, declare it and assign the actual parameter value to it
					// but for array parameters use the original variable if possible
    				// Example: 
    				// void foo(double x[], double *y){}
    				// main(){
    				//  double a[10], b;
    				//  1: foo(a, &b);
    				//  2: foo(&a[0], &b);
    				//  3: foo(&a[1], &b);
    				//  4: foo(a+1, &b);
    				// }
    				// while inlining foo() in main() we should be able to use original array 'a'.
    				// also if y is accessed in foo only in "*y=....;" expression in that case we could use 'b' instead of 'y'

    				ArrayParameterAnalysis arrayParameterAnalysis = new ArrayParameterAnalysis(getProgram(enclosingFunction), ArrayParameterAnalysis.Option.ALLOW_OFFSET);
    				arrayParameterAnalysis.start();
    				
    				for(int k = 0; k < parameters.size(); k++){
    					
						Expression actualArg = args.get(k);
						Symbol array = null;
						
						// handle the "*y=..." case first (note: it is separate from using original array names, but it is dealt here)
						if(actualArg instanceof UnaryExpression && ((UnaryExpression)actualArg).getOperator() == UnaryOperator.ADDRESS_OF) {
							VariableDeclaration pdn = (VariableDeclaration)parameters.get(k);
							Declarator pdd = pdn.getDeclarator(0);
							if(pdd instanceof VariableDeclarator) {
								List l = ((VariableDeclarator) pdd).getSpecifiers();
								if(l.size() == 1 && l.get(0) instanceof PointerSpecifier) {
									Expression uexp = arrayParameterAnalysis.getCompatibleArgument(fc, (VariableDeclarator)pdd);
									if(uexp instanceof UnaryExpression){
										Expression exp = ((UnaryExpression) uexp).getExpression();
										IDExpression oldFormalParam = (IDExpression)(((IDExpression)pdn.getDeclaredIDs().get(0)).clone()); // don't forget to clone
										oldParams.add(oldFormalParam);
					    				actualArg = ((UnaryExpression)actualArg).getExpression();
					    				if(!(actualArg instanceof Identifier)) {
					    					System.out.println("Handle me in InlineExpansion #newid-11");
					    				}
					    				newParamNames.add(exp.toString());
					    				newSymbols.add(((Identifier)actualArg).getSymbol());
					    				Expression offset = uexp.clone();
					    				offset.setParent(null);
				    					offsets.put(oldFormalParam, offset);
					    				//actualArgs.put(exp.toString(), (ArrayAccess)exp);
					    				continue;
									}	
								}
							}
						}
						
						// for cases like '&a[0]'
						if(actualArg instanceof UnaryExpression) {
							actualArg = ((UnaryExpression)actualArg).getExpression();
						}
						// for cases like 'a+1' (or '1+a') (currently we require that the second operand be integeral literal, but ideally it could be a variable, have to handle that)
						else if(actualArg instanceof BinaryExpression) {
							if(((BinaryExpression) actualArg).getLHS() instanceof Identifier && ((BinaryExpression) actualArg).getRHS() instanceof IntegerLiteral) {
								array = ((Identifier)((BinaryExpression) actualArg).getLHS()).getSymbol();
							}
							else if(((BinaryExpression) actualArg).getRHS() instanceof Identifier && ((BinaryExpression) actualArg).getLHS() instanceof IntegerLiteral) {
								array = ((Identifier)((BinaryExpression) actualArg).getRHS()).getSymbol();								
							}
						}
						// handle cases like 'a'
						if(actualArg instanceof IDExpression){

							VariableDeclaration pdn = (VariableDeclaration)parameters.get(k);
							Declarator pdd = pdn.getDeclarator(0);
							if(pdd instanceof VariableDeclarator && pdd.getArraySpecifiers() != null && pdd.getArraySpecifiers().size() > 0){
								Expression exp = arrayParameterAnalysis.getCompatibleArgument(fc, (VariableDeclarator)pdd);
								if(exp != null){
									oldParams.add((IDExpression)(((IDExpression)pdn.getDeclaredIDs().get(0)).clone())); // don't forget to clone
				    				newParamNames.add(exp.toString());
				    				if(!(actualArg instanceof Identifier)) {
				    					System.out.println("Handle me in InlineExpansion #newid-1");
				    				}
				    				newSymbols.add(((Identifier)actualArg).getSymbol());
				    				continue;
								}
							}
						}
						// for cases like '&a[0]'
						if(actualArg instanceof ArrayAccess){
		    				Expression e  = ((ArrayAccess)actualArg).getArrayName();
		    				if(!(e instanceof Identifier)) {
		    					System.out.println("Handle me in InlineExpansion #newid-3");
		    				}
							array = ((Identifier)e).getSymbol();
						}
						// handle the cases '&a[0]' and 'a+1' here
						if(array != null) {
							Declaration dn = (Declaration) parameters.get(k);
							Declarator dd = ((VariableDeclaration)dn).getDeclarator(0);
							if(dd instanceof VariableDeclarator && dd.getArraySpecifiers() != null && dd.getArraySpecifiers().size() > 0){
								Expression exp = arrayParameterAnalysis.getCompatibleArgument(fc, (VariableDeclarator)dd);
								if(exp != null){
									IDExpression oldFormalParam = (IDExpression)(((IDExpression)dn.getDeclaredIDs().get(0)).clone()); // don't forget to clone 
									oldParams.add(oldFormalParam);
				    				// for '&array[0]' ArrayParameterAnalysis returns Identifier 'array' and not ArrayAccess
				    				if(exp instanceof Identifier) {
				    					exp = new ArrayAccess(new Identifier(((Identifier)exp).getSymbol()), new ArrayList<Expression>());
				    				}
				    				else if(exp instanceof CommaExpression) {
				    					List<Traversable> l = ((CommaExpression)exp).getChildren();
				    					if(l.size() != 2) {
				    						System.out.println("Handle me in InlineExpansion #newid-2");
				    					}				    					
				    					exp = (Expression)l.get(0);
					    				if(exp instanceof Identifier) {
					    					exp = new ArrayAccess(new Identifier(((Identifier)exp).getSymbol()), new ArrayList<Expression>());
					    				}
					    				Expression offset = ((Expression)l.get(1)).clone();
					    				offset.setParent(null);
					    				
				    					offsets.put(oldFormalParam, offset);
				    				}
				    				newParamNames.add(exp.toString());
				    				actualArgs.put(exp.toString(), (ArrayAccess)exp);
				    				newSymbols.add(array);
				    				continue;
								}
							}	
							// following is my version, but right now i'm using hansang's method which is more conservative
//							if(dd.getArraySpecifiers() != null && dd.getArraySpecifiers().size() > 0){
//								oldParams.add((IDExpression)(((IDExpression)dn.getDeclaredSymbols().get(0)).clone())); // don't forget to clone
//			    				newParamNames.add(actualArg.toString());
//			    				actualArgs.put(actualArg.toString(), (ArrayAccess)actualArg);
//			    				continue;
//							}
							
						}
						
	    				Declaration d = (Declaration) parameters.get(k);
	    				List<IDExpression> params = d.getDeclaredIDs();
	    				if(d instanceof VariableDeclaration && params.size() == 1){
	    					VariableDeclaration p = (VariableDeclaration)d;
		    				Declarator decl = p.getDeclarator(0); // in case of parameters it seems safe to get 0th declarator, there should be only one
							oldParams.add((IDExpression)(params.get(0).clone())); // don't forget to clone
		    				String paramName = ((NameID)params.get(0)).getName();
		    				// find out a new variable name for this parameter, must be unique
		    				NameID nameId = getUniqueIdentifier(enclosingCompoundStmt, paramPrefix + "_" + functionName2BInlined, paramName);
		    				Identifier id = null;
		    				newParamNames.add(nameId.getName());
		    				// declare the new variable
		    				VariableDeclaration newDeclaration = null;
		    				// it may be using a nested declarator, e.g. int (*param_name)[4]
		    				if(decl instanceof NestedDeclarator){
		    					// clone the original declarator, find the identifier and change its name to new name
		    					NestedDeclarator nd = (NestedDeclarator)decl;
		    					Declarator declarator = nd.getDeclarator();
		    					
		    					if(declarator instanceof VariableDeclarator) {
		    						VariableDeclarator originalVD = (VariableDeclarator)declarator;
		    						
			    					VariableDeclarator vd = new VariableDeclarator(originalVD.getSpecifiers(), nameId, originalVD.getTrailingSpecifiers());
			    					id = new Identifier(vd);
			    					NestedDeclarator newND = new NestedDeclarator(nd.getSpecifiers(), vd, nd.getParameters(), nd.getArraySpecifiers());
	//		    					List<Traversable> children = nd.getChildren();
	//		    					for (int l = 0; l < children.size(); l++) {
	//									if(children.get(l) instanceof VariableDeclarator){
	//										children = ((VariableDeclarator)children.get(l)).getChildren();
	//										for (int l2 = 0; l2 < children.size(); l2++) {
	//											//TODO: can't use setName do something else
	//											if(children.get(l2) instanceof Identifier){
	//												((Identifier)children.get(l2)).setName(id.getName());
	//											}	
	//										}
	//									}
	//								}
			    					newDeclaration = new VariableDeclaration(p.getSpecifiers(), newND);
		    					}
		    					else {
		    						System.err.println("fix me in Inliner... ..#A");
		    					}
		    				}
		    				else if(decl instanceof VariableDeclarator){
		    					VariableDeclarator declarator = (VariableDeclarator)decl;
		    					VariableDeclaration declaration = (VariableDeclaration)declarator.getDeclaration();
			    				// if the parameter is an array declare the new variable as pointer, handle multi-dimensional arrays with care
			    				if(declarator.getArraySpecifiers().size() > 0){
			    					List arraySpecs = declarator.getArraySpecifiers();
			    					if(arraySpecs.size() == 1){
			    						int n = ((ArraySpecifier)arraySpecs.get(0)).getNumDimensions();
			    						// we assume that all the dimensions except the first will have the size mentioned in them
			    						// we are assuming that we are inlining code which is correctly compiled before
										if(n == 1) {
					    					//List<Specifier> newSpecs = declarator.getTypeSpecifiers();
											List<Specifier> newSpecs = new ArrayList<Specifier>(declarator.getSpecifiers());
											newSpecs.add(PointerSpecifier.UNQUALIFIED);
											VariableDeclarator vd = new VariableDeclarator(newSpecs, nameId);
											id = new Identifier(vd);
					    					//newDeclaration = new VariableDeclaration(vd);
											newDeclaration = new VariableDeclaration(declaration.getSpecifiers(), vd);
										}
										else if(n > 1) {
					    					List<Specifier> newSpecs = new ArrayList<Specifier>();
					    					newSpecs.add(PointerSpecifier.UNQUALIFIED);
											VariableDeclarator vd = new VariableDeclarator(newSpecs, nameId);
											id = new Identifier(vd);
					    					List<Specifier> newTrailingSpecs = new ArrayList<Specifier>();
					    					List<Expression> dimensions = new ArrayList<Expression>();
				    						for (int m = 1; m < n; m++) {
				    							dimensions.add((Expression)((ArraySpecifier)arraySpecs.get(0)).getDimension(m).clone());
											}
					    					ArraySpecifier arraySpecifier = new ArraySpecifier(dimensions);
			    							newTrailingSpecs.add(arraySpecifier);
											//NestedDeclarator nd = new NestedDeclarator(declarator.getTypeSpecifiers(), vd, null, newTrailingSpecs);
											//newDeclaration = new VariableDeclaration(nd);
			    							NestedDeclarator nd = new NestedDeclarator(declarator.getSpecifiers(), vd, null, newTrailingSpecs);
                      						newDeclaration = new VariableDeclaration(declaration.getSpecifiers(), nd);
										}
									}
			    					else{
			    						System.err.println("this case not handled in InlineExpansion.java ... 1");
			    					}
			    				}
			    				else{
			    					//VariableDeclarator vd = new VariableDeclarator(declarator.getTypeSpecifiers(), nameId);
			    					VariableDeclarator vd = new VariableDeclarator(declarator.getSpecifiers(), nameId);
			    					id = new Identifier(vd);
			    					//newDeclaration = new VariableDeclaration(vd);	    					
			    					newDeclaration = new VariableDeclaration(declaration.getSpecifiers(), vd);
			    				}
		    				}
		    				else{
		    					System.err.println("unexpected Declarator type, this case not handled in InlineExpansion.java ... 2");
		    				}
		    				if(newDeclaration != null && id != null){
			    				// add the new variable declaration
			    				declarations.addElement(newDeclaration);
			    		        Expression arg = (Expression)fc.getArgument(k).clone(); // don't forget to clone
			    		        // and assign to it the expression that was being passed as a function call, this should be done before the statement with function call 
			    		        ExpressionStatement exprStmt = new ExpressionStatement(new AssignmentExpression(id, AssignmentOperator.NORMAL, arg));
			    		        statements.addElement(exprStmt);
			    		        newSymbols.add(id.getSymbol());
		    				}    
	    				}
	    				else{
	    					// shouldn't get here
	    					System.err.println("FIXME in InlineExpansion.java, wasn't expecting it ... 3");
	    				}
	    			}
					
			        // in the code to be inlined replace the parameters with new variables, but don't declare them in the symbol table
					// of the inlined code as we have already declared them in the surrounding function
					replaceVariableNames(codeToBeInlined, oldParams, newParamNames, actualArgs, offsets, newSymbols, true);
    			}	
				// deal with the global variables that the original function was using, now we need to extern them
				Map<IDExpression, Declaration> usedGlobalVars = getUsedGlobalVariables(function2BInlined, codeToBeInlined, newLocals, newParamNames);
				Iterator<IDExpression> iterator = usedGlobalVars.keySet().iterator();
				while(iterator.hasNext()){
					IDExpression varId = (IDExpression)iterator.next(); //should be safe to cast to Identifier since we checked the equality in getUsedGlobalVariables function
					if(usedGlobalVars.get(varId) instanceof VariableDeclaration){
						VariableDeclaration varDec = (VariableDeclaration)usedGlobalVars.get(varId).clone();
						Declarator decl = getDeclarator(varDec, varId);
						if(decl != null){
							
							List<Specifier> spec = new ArrayList<Specifier>();
							spec.addAll(varDec.getSpecifiers());
							// if we have a static variable declared in the same translation unit, make sure its declaration appears before
							// the function, otherwise move it up.
							// if the global variable is declared in the same translation unit (or header file included in it),
							// do not extern it if it appears before the function.
							TranslationUnit tUnit = getTranslationUnit(enclosingFunction);
							FlatIterator fi = new FlatIterator(tUnit);
							boolean functionTraversed = false;
							boolean externVar = true;
							while(fi.hasNext()){
								obj = fi.next();
								if(obj instanceof Procedure && enclosingFunction.equals(obj)){
									functionTraversed = true;
								}
								if(obj instanceof VariableDeclaration && usedGlobalVars.get(varId).equals(obj)){
									if(functionTraversed) {
										if(spec.contains(Specifier.STATIC)) {
											// move the declaration up
											removedGlobalDeclarations.addElement((VariableDeclaration)obj);
											addedGlobalDeclarations.addElement((VariableDeclaration)obj);
										}
									}
									else {
										externVar = false;
									}	
									break;
								}
							}
							// don't extern the static variable, and the global variable if it is declared in the same translation unit
							if(spec.contains(Specifier.STATIC) || !externVar){
								continue;
							}
							
							if(!spec.contains(Specifier.EXTERN)) {
								spec.add(0, Specifier.EXTERN);
							}	
							
							// if the surrounding function has a local variable or parameter having the same name as the used global variable
							// in the body of the function to be inlined, we need to rename that local variable/parameter and its usage
							// in the surrounding function.
							
							Set<Symbol> symbols = enclosingFunctionBody.getSymbols();
							Set<Symbol> symbols2 = enclosingFunction.getSymbols();
					        List<Symbol> locals2 = new LinkedList<Symbol>();
					        List<Symbol> params2 = new LinkedList<Symbol>();
					        List<IDExpression> localsNames = new LinkedList<IDExpression>();
					        List<String> newLocalsNames = new LinkedList<String>();
					        List<IDExpression> paramsNames = new LinkedList<IDExpression>();
					        List<String> newParamsNames = new LinkedList<String>();
					        boolean extern = true;
					        for (Iterator iterator2 = symbols.iterator(); iterator2.hasNext();) {
								Symbol symbol = (Symbol) iterator2.next();
								if(symbol.getSymbolName().equals(varId.getName())) {
									// if it is already externed we won't do it again!	
									if(symbol instanceof VariableDeclarator) {
										Traversable tr = ((VariableDeclarator)symbol).getParent();
										if(tr instanceof VariableDeclaration) {
											List<Specifier> sp = ((VariableDeclaration)tr).getSpecifiers();
											if(sp.contains(Specifier.EXTERN))
												extern = false;
										}
									}
									if(extern) {
										localsNames.add(varId);
										locals2.add(symbol);
									}	
									extern = false;
								}	
							}
							for (Iterator iterator2 = symbols2.iterator(); iterator2.hasNext();) {
								Symbol symbol = (Symbol) iterator2.next();
								if(symbol.getSymbolName().equals(varId.getName())) {
									paramsNames.add(varId);
									params2.add(symbol);
									extern = false;
								}	
							}
					        // come up with unique names for new variables and labels.
					        for (int k = 0; k < localsNames.size(); k++) {
								newLocalsNames.add(getUniqueIdentifier(enclosingFunctionBody, "" + enclosingFunctionName, localsNames.get(k).toString()).getName());
							}
					        // come up with unique names for new params and labels.
					        for (int k = 0; k < paramsNames.size(); k++) {
								newParamsNames.add(getUniqueIdentifier(enclosingFunctionBody, "" + enclosingFunctionName, paramsNames.get(k).toString()).getName());
							}
					        // rename local variables
					        for (int k = 0; k < locals2.size(); k++) {
								locals2.get(k).setName(newLocalsNames.get(k));
							}
					        // rename params (we need to rename them differently, ProcedureDeclarator's parent is null, so we can't use the normal Symbol.setName()
					        // function, we have to use the SymbolTools.setSymbolName() utility function for this
					        for (int k = 0; k < params2.size(); k++) {
					        	SymbolTools.setSymbolName(params2.get(k), newParamsNames.get(k), enclosingFunction);
							}
							// handle labels and name changing in annotations
					        if(localsNames.size() > 0) {
					        	replaceVariableNames(enclosingFunctionBody, localsNames, newLocalsNames, new HashMap<String, ArrayAccess>(), new HashMap<IDExpression, Expression>(), new LinkedList<Symbol>(), false);
					        }
					        if(paramsNames.size() > 0) {
					        	replaceVariableNames(enclosingFunctionBody, paramsNames, newParamsNames, new HashMap<String, ArrayAccess>(), new HashMap<IDExpression, Expression>(), new LinkedList<Symbol>(), false);
					        }
					        if(!extern) {
					        	continue;
					        }
							NameID name = new NameID(varId.getName());
							Declarator newDecl = null;
							if(decl instanceof NestedDeclarator){
								VariableDeclarator originalVD = (VariableDeclarator)((NestedDeclarator)decl).getDeclarator();
								VariableDeclarator vd = new VariableDeclarator(originalVD.getSpecifiers(), name, originalVD.getTrailingSpecifiers());
								varId = new Identifier(vd);
		    					newDecl = new NestedDeclarator(((NestedDeclarator)decl).getSpecifiers(), vd, ((NestedDeclarator)decl).getParameters(), ((NestedDeclarator)decl).getArraySpecifiers());

							}
							else if(decl instanceof VariableDeclarator){
								newDecl = new VariableDeclarator(decl.getSpecifiers(), name, ((VariableDeclarator)decl).getTrailingSpecifiers());
								varId = new Identifier((VariableDeclarator)newDecl);
							}
							else {
								System.err.println("fix me in inliner #B");
							}
							if(newDecl != null) {
								VariableDeclaration newExternDeclaration = new VariableDeclaration(spec, newDecl);
								declarations.addElement(newExternDeclaration);
								//TODO: enclosing compound statement may already has such extern declaration, would be nice to avoid
								// multiple externs of the same variable.
							}	
						}	
					}
				}
				// if some function calls in the code to be inlined are not inlined, we need to make sure that we have the declarations
				// for those function calls, if not we got to get them from the callee's header files
				// if they are using user specifier types in the parameters, we need to get their declarations as well
				// and declare them in caller's translation. while doing so, we should follow the chains of typedefs as well the struct
				// elements. However, in case of a name conflict, we would give up and not inline.
				dfi = new DepthFirstIterator(codeToBeInlined);
				TranslationUnit callerUnit = getTranslationUnit(enclosingFunction);
				TranslationUnit calleeUnit = getTranslationUnit(function2BInlined);
				
				while(dfi.hasNext()){
					if((obj = dfi.next()) instanceof FunctionCall){
						FunctionCall functionCall = (FunctionCall)obj;
						// if we don't have the declaration, get it from the callee's translation unit and declare it
						// but if the arguments are typedefed we don't want to create problems, so we won't inline
						if(!isDeclarationAvailable(functionCall, callerUnit)){
							Object o = getFunctionDeclaration(functionCall, calleeUnit);
							if(o != null){
								VariableDeclaration d = null;
								if(o instanceof VariableDeclaration){
									d = (VariableDeclaration)((VariableDeclaration)o).clone();
								}
								if(o instanceof Procedure){
									Procedure p = (Procedure)o;
									d = new VariableDeclaration(p.getReturnType(), (Declarator)p.getDeclarator().clone());
								}

								if(d.getNumDeclarators() != 1 && !(d.getDeclarator(0) instanceof ProcedureDeclarator))
									System.err.println("unexpected..needs to be fixed in InlineExpansion.java ... 4");
								
								ProcedureDeclarator pd = (ProcedureDeclarator)d.getDeclarator(0);
								List<Declaration> p = pd.getParameters();
								
					CHECK_PARAMS: for (int l = 0; l < p.size(); l++) {
									if(!(p.get(l) instanceof VariableDeclaration))
										System.err.println("unexpected .. needs to be fixed in InlineExpansion.java ... 5");
										
									List<Specifier> specs = ((VariableDeclaration)p.get(l)).getSpecifiers();
									Vector<UserSpecifier> toBeResolved = new Vector<UserSpecifier>();
									for (int m = 0; m < specs.size(); m++) {
										if(specs.get(m) instanceof UserSpecifier){
											toBeResolved.addElement((UserSpecifier)specs.get(m));
										}
									}
										
									while(!toBeResolved.isEmpty()){
										UserSpecifier userSpec = toBeResolved.remove(0);
										IDExpression userSpecName = userSpec.getIDExpression();
										Declaration original = null;
										// check if it is in the callee's symbol table
										if((original = calleeUnit.findSymbol(userSpecName)) != null){
											
											// we might have already decided to add this declaration to the caller's symbol table
											if(addedGlobalDeclarations.contains(original))
												continue;
											
											// it might be a typedef
											if(original instanceof VariableDeclaration){
												VariableDeclaration vd = (VariableDeclaration)original;
												Vector<UserSpecifier> us = new Vector<UserSpecifier>();
												boolean isTypedef = false;
												for (Specifier spec : vd.getSpecifiers()) {
													if(spec.equals(Specifier.TYPEDEF)){
														isTypedef = true;
													}
													else if(spec instanceof UserSpecifier){
														us.addElement((UserSpecifier)spec);
													}
												}
												if(isTypedef){
													// check the declaration in the caller's symbol table
													Declaration callerDecl = null;
													boolean addDeclaration = true;
													if((callerDecl = callerUnit.findSymbol(userSpecName)) != null){
														// if we have a declaration with this name but it doesn't match we got a name conflict, we should give up and not online
														if(callerDecl instanceof VariableDeclaration && callerDecl.toString().equals(original.toString())){
//															// if it matches we got to remove the original one if it is in the .c file and add the new one in the beginning
//															if(callerUnit.getTable().containsKey(userSpecName))
//																removedGlobalDeclarations.addElement(callerDecl);
//															else
															addDeclaration = false;
														}
														else{
															if(makeLog){
																System.out.println("function \'" + functionName2BInlined +  "\' in the following function call inside function \'" + enclosingFunctionName + 
																		"\' can't be inlined because the declaration of one of the non-inlined functions in it contains parameter(s) whose type " +
																		"clashes with type(s) in the caller translation unit, so we can't inline");
																System.out.println(statementWithFunctionCall.toString());
												    		}	
															inline = false;
															break CHECK_PARAMS;
														}
													}
													
													if(addDeclaration){
														// the declaration should be added
														addedGlobalDeclarations.addElement(vd);
														// if it involves other user specified types, we need to deal with those too
														toBeResolved.addAll(us);
													}	
												}
												else{
													System.err.println("handle this case..in InlineExpansion.java ... 6");
												}
												
											}
											// it might be a struct
											else if(original instanceof ClassDeclaration){
												ClassDeclaration cd = (ClassDeclaration)original;
												// check if the caller already has such declaration
												Declaration callerDecl = null;
												boolean addDeclaration = true;
												if((callerDecl = callerUnit.findSymbol(userSpecName)) != null){
													// if they don't match, we got a name conflict, we should give up and not inline
													if(callerDecl instanceof ClassDeclaration && callerDecl.toString().equals(original.toString())){
//														// if it matches and is in the .c file we got to remove the original one and add the new one in the beginning
//														if(callerUnit.getTable().containsKey(userSpecName))
//															removedGlobalDeclarations.addElement(callerDecl);
//														else
														addDeclaration = false;
													}
													else{
														if(makeLog){
															System.out.println("function \'" + functionName2BInlined +  "\' in the following function call inside function \'" + enclosingFunctionName + 
																	"\' can't be inlined because the declaration of one of the non-inlined functions in it contains parameter(s) whose type " +
																	"clashes with type(s) in the caller translation unit, so we can't inline");
															System.out.println(statementWithFunctionCall.toString());
											    		}	
														inline = false;
														break CHECK_PARAMS;
													}
												}
												
												if(addDeclaration){
													// the declaration should be added
													addedGlobalDeclarations.addElement(cd);
	
													// if declarations inside the struct use other structs or typedefs, handle them here
													for (Traversable traversable : cd.getChildren()) {
														if(traversable instanceof DeclarationStatement){
															DeclarationStatement ds = (DeclarationStatement)traversable;
															if(!(ds.getDeclaration() instanceof VariableDeclaration))
																System.err.println("not handled ... handle in InlineExpression.java ... 7");
															
															VariableDeclaration vd = (VariableDeclaration)ds.getDeclaration();
															for (Specifier spec : vd.getSpecifiers()) {
																if(spec instanceof UserSpecifier){
																	toBeResolved.addElement((UserSpecifier)spec);
																}
															}
														}
														else{
															System.err.println("not handled ... handle in InlineExpansion.java ... 8");
														}	
													}
												}	
											}
										}
										else{
											if(makeLog){
												System.out.println("function \'" + functionName2BInlined +  "\' in the following function call inside function \'" + enclosingFunctionName + 
														"\' can't be inlined because the declaration of one of the non-inlined functions in it contains parameter(s) whose type is unknown, so we can't extern it");
												System.out.println(statementWithFunctionCall.toString());
								    		}	
											inline = false;
											break CHECK_PARAMS;
										}
									}
								}
								if(inline){
									// add the declaration itself, in the beginning of the caller's translation unit
									callerUnit.addDeclarationFirst(d);
								}	
							}
							else{
								if(makeLog){
									System.out.println("function \'" + functionName2BInlined +  "\' in the following function call inside function \'" + enclosingFunctionName + 
											"\' can't be inlined because we couldn't find declaration of one of the non-inlined functions in it");
									System.out.println(statementWithFunctionCall.toString());
					    		}	
								inline = false;
							}
						}
					}
				}
				
				if(inline){
					if(!inlinedFunctions.contains(function2BInlined)) {
						inlinedFunctions.add(function2BInlined);
					}	
					// modify the body of the function and the translation unit with the inlined code and necessary declarations and assignment statements
					for (Declaration d : declarations) {
						if(statementWithFunctionCall instanceof DeclarationStatement) {
							enclosingCompoundStmt.addDeclarationBefore(((DeclarationStatement)statementWithFunctionCall).getDeclaration(), d);
						}
						else {
							enclosingCompoundStmt.addDeclaration(d);							
						}						
					}
					for (Statement s : statements) {
						enclosingCompoundStmt.addStatementBefore(statementWithFunctionCall, s);
					}
					for (Declaration d : removedGlobalDeclarations) {
						callerUnit.removeChild(d);
					}
					for (Declaration d : addedGlobalDeclarations) {
						// these should go to the beginning of the file, but clone them first as we haven't done that earlier
						callerUnit.addDeclarationFirst((Declaration)d.clone());
					}
					
					// deal with the return statement
					ReturnStatement returnStmt = getReturnStatement(codeToBeInlined);
					Identifier returnVarId = null;
					if(returnStmt != null){
						if(returnStmt.getExpression() != null){
							// store the return expression in new unique variable defined in the scope of the surrounding function and remove the return statement
							Expression returnExpr = (Expression)returnStmt.getExpression().clone();
							NameID returnVarNameId = getUniqueIdentifier(enclosingCompoundStmt, resultPrefix, function2BInlined.getSymbolName());
							//VariableDeclarator vd = new VariableDeclarator(returnVarNameId);
							VariableDeclarator vd = new VariableDeclarator(function2BInlined.getDeclarator().getSpecifiers(), returnVarNameId);
							returnVarId = new Identifier(vd);
							List<Specifier> returnType = function2BInlined.getReturnType();
							returnType.removeAll(function2BInlined.getDeclarator().getSpecifiers()); // removes declarator specs already included in vd
							// for a static function return type also has static, we don't want that otherwise our result variable would be
							// declared as static
							returnType.remove(Specifier.STATIC);							
							enclosingCompoundStmt.addDeclaration(new VariableDeclaration(returnType, vd));
							ExpressionStatement exprStmt = new ExpressionStatement(new AssignmentExpression(returnVarId, AssignmentOperator.NORMAL, returnExpr));
							returnStmt.swapWith(exprStmt);
						}
						else{
							NullStatement nullStmt = new NullStatement();
							returnStmt.swapWith(nullStmt);
						}
					}
				
					// add the comment
					enclosingCompoundStmt.addStatementBefore(statementWithFunctionCall, new AnnotationStatement(new CommentAnnotation("inlining function " + functionName2BInlined + " in the body of function " + enclosingFunctionName)));
					// include the inlined code before the statement with function call
					enclosingCompoundStmt.addStatementBefore(statementWithFunctionCall, codeToBeInlined);
					
					// deal with the function call in the statement 
					if(returnStmt != null && returnVarId != null){
						// replace the function call with the variable that holds the return value  
						fc.swapWith(returnVarId.clone());
					}
					else{ 
						// it is a void function
						// just remove the function call
						//TODO: removeChild can't be used ... do something else
						//((Statement)fc.getParent()).removeChild(fc);
						Statement original = (Statement)fc.getParent();
						// if it is just one function call (and for void function calls it should be the case)
						if(original.getChildren().size() == 1 && original.getParent() instanceof CompoundStatement){
							((CompoundStatement)original.getParent()).removeStatement(original);
						}
						else{
						    int index = Tools.indexByReference(original.getChildren(), fc);
						    if (index != -1){
						    	original.setChild(index, new NullStatement());
						    }    
						    else{
						    	System.err.println("Fix me in InlineExpansion.java, couldn't remove function call");
						    }
						}    
					}
				}	
    		}
	    }	
	}
	/**
	 * returns a unique identifier for the given compound statement based on the provided hints
	 * depending upon the fullyQualifiedName flag, this method returns names in two different formats
	 * when the flag is true, it returns "_[prefix]_[suffix]", in case of conflict it adds number to it, i.e. "_[prefix]_x[suffix]"
	 * where x is 2,3,4,...
	 * when the flag is false, it does not append or prepend anything unless in case of conflict in which case it would append a number
	 * Here are the two examples:
	 * a) _local_func1__local_Func2__param_Func3_array // when fullyQualifiedName flag is set to true
	 * b) array // when the flag is false
	 * The length of the variable name is also kept into account, to make sure it does not exceed the maximum allowed limit
	 * @param compoundStmt - the compound statement (may be function body) for which unique identifier is to be sought
	 * @param prefix - prefix hint
	 * @param suffix - suffix hint
	 * @return - the unique identifier
	 */
	private NameID getUniqueIdentifier(SymbolTable compoundStmt, String prefix, String suffix){
		int i = 1;
		String newName = "_" + prefix + "_" + suffix;
//		if(!fullyQualifiedName && prefix.startsWith(localVarPrefix) && suffix.indexOf(localVarPrefix) != -1)
//			newName = "_" + suffix;
		if(!fullyQualifiedName)
			newName = suffix;
		newName = adjustLength(newName);
		NameID id = new NameID(newName);
		while(compoundStmt.findSymbol(id) != null){
			newName = "_" + prefix + "_" + (++i) + "_" + suffix;
//			if(!fullyQualifiedName && prefix.startsWith(localVarPrefix) && suffix.indexOf(localVarPrefix) != -1)
//				newName = "_" + i + suffix;
			if(!fullyQualifiedName)
				newName = suffix + i;
			newName = adjustLength(newName);
			id = new NameID(newName);
		}
		return id;
	}
	/**
	 * returns a unique label for the given function based on the provided hints
	 * depending upon the fullyQualifiedName flag, this method returns names in two different formats
	 * when the flag is true, it returns "_[prefix]_[suffix]", in case of conflict it adds number to it, i.e. "_[prefix]_x[suffix]"
	 * where x is 2,3,4,...
	 * when the flag is false, it does not add label prefix with the suffix if it already has label prefix in front of it
	 * all it appends in this case is an underscore
	 * Here are the two examples:
	 * The length of the variable name is also kept into account, to make sure it does not exceed the maximum allowed limit
	 * @param proc - the function
	 * @param prefix - prefix hint
	 * @param suffix - suffix hint
	 * @return - the unique identifier
	 */
	private NameID getUniqueLabel(Procedure proc, String prefix, String suffix){
		int i = 1;
		String newName = "_" + prefix + "_" + suffix;
		if(!fullyQualifiedName && suffix.startsWith(labelPrefix))
			newName = "_" + suffix;
		newName = adjustLength(newName);
		NameID newLabel = new NameID(newName);
		CompoundStatement body = proc.getBody();
		DepthFirstIterator dfi = new DepthFirstIterator(body);
		Vector<NameID> labels = new Vector<NameID>();
		Object next = null;
		while(dfi.hasNext()){
			if((next = dfi.next()) instanceof Label)
				labels.addElement((NameID)((Label)next).getName());
		}
		while(labels.contains(newLabel)){
			newName = "_" + prefix + "_" + (++i) + "_" + suffix;
			if(!fullyQualifiedName && prefix.startsWith(localVarPrefix) && suffix.indexOf(localVarPrefix) != -1)
				newName = "_" + i + suffix;
			newName = adjustLength(newName);
			newLabel = new NameID(newName);
		}
		return newLabel;
	}
	/**
	 * makes sure that the given name is not longer than maximum allowed length. If it is, it is renamed. If there are underscores
	 * in the beginning of the name (e.g. when the fullyQualifiedName flag is false), they are removed and a string "_t_" is appended
	 * to mean that the name has been truncated. Otherwise the last half is chopped off.  
	 * @param name - name of a variable
	 * @return - variable name after making sure it is not longer than the maximum allowed length, returned name may be different from the 
	 *           one passed
	 */
	private String adjustLength(String name) {
		if(name.length() < maxVarLength)
			return name;
		for (int i = 0; i < name.length(); i++) {
			if(Character.isLetter(name.charAt(i))){
				String temp = name.substring(i);
				if(temp.length() + 3 < maxVarLength)
					return "_t_" + temp;
				else
					break;
			}	
		}
		return name.substring(0, maxVarLength/2);
	}
	/**
	 * returns the return statement in the body of a function specified by the compound statement
	 * Note: shouldn't it be moved to Procedure class?
	 * @param functionBody - body of the function 
	 * @return - the return statement or null if there is none
	 */
	private ReturnStatement getReturnStatement(CompoundStatement functionBody){
		DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
		Object next = null;
		while(dfi.hasNext()){
			if( (next = dfi.next()) instanceof ReturnStatement){
				return (ReturnStatement)next;
			}
		}
		return null;
	}
	/**
	 * renames local variables
	 * @param functionBody
	 * @param oldVars
	 * @param newVarNames
	 */
	private void renameLocals(CompoundStatement functionBody, List<IDExpression> oldVars, List<String> newVarNames){
		if(oldVars.size() == newVarNames.size()){
			for (int i = 0; i < oldVars.size(); i++) {
//				Tools.renameSymbol(functionBody, oldVars.get(i).toString(), newVarNames.get(i));
//				if(true)
//					continue;
//				Declaration d = functionBody.getTable().remove(oldVars.get(i));
				String oldVarName = oldVars.get(i).toString();//getSymbol().getSymbolName(); in case of label symbol is null, so use toString instead
				DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
				Object next = null;
				while(dfi.hasNext()){
					if( (next = dfi.next()) instanceof IDExpression){
						if(((IDExpression)next).getName().equals(oldVarName)){
							Traversable t = ((IDExpression)next).getParent();
							if(t instanceof Symbol){
								//renameVarInDeclarator((Declarator)t, (Identifier)next, newVarNames.get(i), args);
								((Symbol)t).setName(newVarNames.get(i));
							}
						}
					}
				}
			}
		}
	}	

	/**
	 * replaces variable names, is usually called for parameters
	 * @param functionBody - the function body
	 * @param oldVars - list of old variables (IDEXpression instances)
	 * @param newVarNames - list of new variable names
	 * @param args - map containing actual ArrayAccess instances
	 * @param offsets - map containing offsets (e.g. n, for the case of &a[n])
	 * @param symbols - symbols to which new identifiers should point to
	 * @param isParam - if it is called to rename parameters
	 */
	private void replaceVariableNames(CompoundStatement functionBody, List<IDExpression> oldVars, List<String> newVarNames, HashMap<String, ArrayAccess> args, HashMap<IDExpression, Expression> offsets, List<Symbol> symbols, boolean isParam){
		
		if(oldVars.size() == newVarNames.size()){
			
			// if we are using actual array variables from the caller we may run into problem
			// e.g. if the caller was passing arrays x and y and callee was using them as y and x
			// we need to use a temporary to carefully handle it, otherwise we'll end up using either x, x or y, y for x and y.
			// Note: this happens in ft benchmark (when cfftz calls fttz2) 
			String dummyPrefix = "$#@%#@";
			if(isParam) {
				List<IDExpression> oldVarsExtra = new ArrayList<IDExpression>();
				List<String> newVarNamesExtra = new ArrayList<String>();
				List<Symbol> symbolsExtra = new ArrayList<Symbol>();
				for (int i = 0; i < oldVars.size(); i++) {
					IDExpression oldVar = oldVars.get(i);
					String oldVarName = oldVar.toString();
					if(!oldVarName.equals(newVarNames.get(i))) {
						if(newVarNames.contains(oldVarName)) {
							String newName = newVarNames.get(i);
							String newDummyName = dummyPrefix + i + newName;
							newVarNames.set(i, newDummyName);
							NameID oldVarTemp = new NameID(newDummyName);
							oldVarsExtra.add(oldVarTemp);
							newVarNamesExtra.add(newName);
							Expression exp = offsets.get(oldVar);
							if(exp != null) {
								offsets.remove(oldVar);
								offsets.put(oldVarTemp, exp);
							}
							symbolsExtra.add(symbols.get(i));
						}
					}
				}
				oldVars.addAll(oldVarsExtra);
				newVarNames.addAll(newVarNamesExtra);
				symbols.addAll(symbolsExtra);
			}	
			for (int i = 0; i < oldVars.size(); i++) {
				String oldVarName = oldVars.get(i).toString();//getSymbol().getSymbolName(); in case of label symbol is null, so use toString instead
				DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
				Object next = null;
				while(dfi.hasNext()){
					if( (next = dfi.next()) instanceof IDExpression && isParam){
						if(((IDExpression)next).getName().equals(oldVarName)){
							Traversable t = ((IDExpression)next).getParent();
							boolean ok = true;

							// first handle the "*y=..." case (where y is an integer pointer and it is only used in this expression inside the function) 
							if(t instanceof UnaryExpression) {
								if(((UnaryExpression)t).getOperator() == UnaryOperator.DEREFERENCE) {
									Expression offset = offsets.get(oldVars.get(i));
									if(offset instanceof UnaryExpression) {
										Expression exp = ((UnaryExpression) offset).getExpression().clone();
										exp.setParent(null);
										((UnaryExpression)t).swapWith(exp);
										ok = false;
									}
								}
							}
							// a struct member may have the same name as the param
							if(t instanceof AccessExpression) {
								Expression ex = ((AccessExpression) t).getLHS();
								List<Expression> matches = ex.findExpression((IDExpression)next);
								if(matches.size() == 0) {
									ok = false;
								}
							}
							if(ok){ 
								try{
									if(t != null && !newVarNames.get(i).startsWith(dummyPrefix)){
										int index = Tools.indexByReference(t.getChildren(), next);
										if(index != -1){
											if(newVarNames.get(i).indexOf('[') >= 0 && t instanceof ArrayAccess){
												List<Expression> indices = ((ArrayAccess)t).getIndices();
												ArrayAccess original = args.get(newVarNames.get(i));
												if(original != null){
													List<Expression> callerIndices = original.getIndices();
													List<Expression> newIndices = new ArrayList<Expression>();
													// for &a[n] case, check if we need to add offset n
													Expression offset = offsets.get(oldVars.get(i));
													for(Expression ind : callerIndices){
														ind.setParent(null);
														newIndices.add(ind.clone());
													}
													for(Expression ind : indices){
														if(offset != null) {
															ind.setParent(null);
															offset.setParent(null);
															BinaryExpression bExp = new BinaryExpression(ind, BinaryOperator.ADD, offset);
															newIndices.add(bExp);
															// do it for the first index only
															offset = null;
														}
														else {
															ind.setParent(null);
															newIndices.add(ind);
														}	
													}
													
													((ArrayAccess)t).setIndices(newIndices);
												}
											}
										}
										else{
											System.out.println("name could not be changed ... check it");
										}
									}	
								}
								catch(UnsupportedOperationException ex){
									System.out.println("handle me in Inliner...." + t.getClass().getName() + "does not support setChild()");
								}
								if(newVarNames.get(i).startsWith(dummyPrefix)) {
									((IDExpression)next).swapWith(new NameID(newVarNames.get(i)));
								}
								else {
									((IDExpression)next).swapWith(new Identifier(symbols.get(i)));
								}	
							}	
						}	
					}
					// also change the variable names inside annotations
					else if(next instanceof Annotatable){
						// first deal with the labels
						if(next instanceof Label){
							if(((Label)next).getName().getName().equals(oldVarName)){
								((Label)next).setName(new NameID(newVarNames.get(i)));
							}
						}
						// change the labels in the goto statements
						else if(next instanceof GotoStatement){
							Expression exp = ((GotoStatement)next).getExpression();
							if(exp != null && exp.toString() != null && exp.toString().equals(oldVarName)){
								((GotoStatement)next).setExpression(new NameID(newVarNames.get(i)));
							}
						}
						// modify the annotations as well
						List<Annotation> annotations = ((Annotatable)next).getAnnotations();
						if(annotations != null){
							for(Annotation a : annotations){
								Iterator<String> iter = a.keySet().iterator();
								while(iter.hasNext()){
									String key = iter.next();
									Object val = a.get(key);
									
									if(val instanceof String && ((String)val).equals(oldVarName)){
										// replace the value
										a.put(key, newVarNames.get(i));
									}
									else if(val instanceof Collection){
										replaceNameInCollection((Collection)val, oldVarName, newVarNames.get(i));
									}
									else if(val instanceof Map){
										replaceNameInMap((Map)val, oldVarName, newVarNames.get(i));
									}
								}
							}	
						}
					}
				}
			}
		}
	}
	
// pre 08/30/2010 version	
//	/**
//	 * replaces variable names, also removes the old symbols from the symbol table
//	 * @param functionBody - the function body
//	 * @param oldVars - list of old variables (IDEXpression instances)
//	 * @param newVarNames - list of new variable names
//	 * @param addVarsInTable - adds the new variables in the symbol table if this argument is true
//	 * @param args - map containing actual ArrayAccess instances
//	 */
//	private void replaceVariableNames(CompoundStatement functionBody, List<IDExpression> oldVars, List<String> newVarNames, boolean addVarsInTable, HashMap<String, ArrayAccess> args){
//		if(oldVars.size() == newVarNames.size()){
//			for (int i = 0; i < oldVars.size(); i++) {
////				Tools.renameSymbol(functionBody, oldVars.get(i).toString(), newVarNames.get(i));
////				if(true)
////					continue;
////				Declaration d = functionBody.getTable().remove(oldVars.get(i));
//				String oldVarName = oldVars.get(i).toString();//getSymbol().getSymbolName(); in case of label symbol is null, so use toString instead
//				DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
//				Object next = null;
//				while(dfi.hasNext()){
//					if( (next = dfi.next()) instanceof IDExpression){
//						if(((IDExpression)next).getName().equals(oldVarName)){
//							Traversable t = ((IDExpression)next).getParent();
//							try{
//								if(t instanceof Symbol){
//									//renameVarInDeclarator((Declarator)t, (Identifier)next, newVarNames.get(i), args);
//									((Symbol)t).setName(newVarNames.get(i));
//								}
//								else if(t != null){
//									int index = Tools.indexByReference(t.getChildren(), next);
//									if(index != -1){
//										if(newVarNames.get(i).indexOf('[') >= 0 && t instanceof ArrayAccess){
//											List<Expression> indices = ((ArrayAccess)t).getIndices();
//											ArrayAccess original = args.get(newVarNames.get(i));
//											if(original != null){
//												List<Expression> callerIndices = original.getIndices();
//												List<Expression> newIndices = new ArrayList<Expression>();
//												for(Expression ind : callerIndices){
//													newIndices.add(ind.clone());
//												}
//												for(Expression ind : indices){
//													ind.setParent(null);
//													newIndices.add(ind);
//												}
//												((ArrayAccess)t).setIndices(newIndices);
//												t.setChild(index, original.getArrayName().clone());
//											}
//											else{
//												//TODO: use of setChild is dangerous and NameID won't make into symbol table
//												t.setChild(index, new NameID(newVarNames.get(i)));
//											}	
//										}
//										else{
//											//TODO: use of setChild is dangerous and NameID won't make into symbol table
//											t.setChild(index, new NameID(newVarNames.get(i)));
//										}
//									}
//									else{
//										System.out.println("name could not be changed ... check it");
//									}
//								}	
//							}
//							catch(UnsupportedOperationException ex){
//								System.out.println("handle me in Inliner...." + t.getClass().getName() + "does not support setChild()");
//							}
//						}	
//					}
//					// also change the variable names inside annotations
//					else if(next instanceof Annotatable){
//						// first deal with the labels
//						if(next instanceof Label){
//							if(((Label)next).getName().getName().equals(oldVarName)){
//								((Label)next).setName(new NameID(newVarNames.get(i)));
//							}
//						}
//						// change the labels in the goto statements
//						else if(next instanceof GotoStatement){
//							Expression exp = ((GotoStatement)next).getExpression();
//							if(exp != null && exp.toString() != null && exp.toString().equals(oldVarName)){
//								((GotoStatement)next).setExpression(new NameID(newVarNames.get(i)));
//							}
//						}
//						// modify the annotations as well
//						List<Annotation> annotations = ((Annotatable)next).getAnnotations();
//						if(annotations != null){
//							for(Annotation a : annotations){
//								Iterator<String> iter = a.keySet().iterator();
//								while(iter.hasNext()){
//									String key = iter.next();
//									Object val = a.get(key);
//									
//									if(val instanceof String && ((String)val).equals(oldVarName)){
//										// replace the value
//										a.put(key, newVarNames.get(i));
//									}
//									else if(val instanceof Collection){
//										replaceNameInCollection((Collection)val, oldVarName, newVarNames.get(i));
//									}
//									else if(val instanceof Map){
//										replaceNameInMap((Map)val, oldVarName, newVarNames.get(i));
//									}
//								}
//							}	
//						}
//					}
//				}
//			}
//			if(addVarsInTable){
//				DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
//				Object next = null;
//				while(dfi.hasNext()){
//					if( (next = dfi.next()) instanceof Declaration){
//						SymbolTools.addSymbols(functionBody, (Declaration)next);
//					}
//				}
//			}
//		}
//	}
	// Following is not needed, now we can call setName() on symbols
//	/**
//	 * renames variable in the given declarator
//	 * @param d - the declarator
//	 * @param id - identifier for the variable that needs to be renamed
//	 * @param newVarName - new name for the variable
//	 * @param args - 
//	 */
//	private void renameVarInDeclarator(Declarator d, Identifier id, String newVarName, Map<String, ArrayAccess>args){
//		if(d instanceof VariableDeclarator){
//			Declarator oldD = (VariableDeclarator)d;
//			Declarator newD = new VariableDeclarator(oldD.getSpecifiers(), new Identifier(newVarName), oldD.getArraySpecifiers());
//			newD.setInitializer(oldD.getInitializer());
//			Traversable t;
//			while( (t = oldD.getParent()) instanceof NestedDeclarator){
//				oldD = (NestedDeclarator)t;
//				newD = new NestedDeclarator(oldD.getSpecifiers(), newD, oldD.getParameters(), oldD.getArraySpecifiers());
//			}
//			int index = Tools.indexByReference(t.getChildren(), oldD);
//			SymbolTable table = getSymbolTable(t);
//			if(table != null){
//				if(index != -1 && t instanceof Declaration){
//					SymbolTools.removeSymbols(table, (Declaration)t);
//					t.setChild(index, newD);
//					SymbolTools.addSymbols(table, (Declaration)t);
//				}
//				
//			}
//			else{
//				System.out.println("fix me in inliner");
//			}
//		}
//	}
	
	private SymbolTable getSymbolTable(Traversable t){
		while(t != null && !(t instanceof SymbolTable))
			t = t.getParent();
		
		if(t instanceof SymbolTable)
			return (SymbolTable)t;
		
		return null;
	}
	
	/**
	 * replaces the given old name in the given collection with the new name, if the values are themselves collections or maps 
	 * recursively handles them also
	 */
	private void replaceNameInCollection(Collection c, String name, String newName){
		Iterator iter = ((Collection)c).iterator();
		while(iter.hasNext()){
			Object val = iter.next();
			if(val instanceof String && ((String)val).equals(name)){
				c.remove(val);
				c.add(newName);
				break;
			}
			else if(val instanceof Collection){
				replaceNameInCollection((Collection)val, name, newName);
			}
			else if(val instanceof Map){
				replaceNameInMap((Map)val, name, newName);
			}
		}
	}
	/**
	 * replaces the given old name in the values of given map with the new name, if the values are themselves collections or maps 
	 * recursively handles them also
	 */
	private void replaceNameInMap(Map m, String name, String newName){
		Iterator<String> iter = m.keySet().iterator();
		while(iter.hasNext()){
			String key = iter.next();
			Object val = m.get(key);
			if(val instanceof String && ((String)val).equals(name)){
				// replace the value
				m.put(key, newName);
			}
			else if(val instanceof Collection){
				replaceNameInCollection((Collection)val, name, newName);
			}
			else if(val instanceof Map){
				replaceNameInMap((Map)val, name, newName);
			}
		}
	}
	/**
	 * For a given function call, this method returns the overall statement that encloses this call 
	 * @param call - the function call
	 * @return - the maximally complete statement which encloses the given function call
	 */
	private Statement getStatement(FunctionCall call){
		Traversable t = call;
		Traversable parent;
		while( (parent = t.getParent()) != null && !(parent instanceof CompoundStatement)){
			t = parent;
		}
		return (Statement)t;
	}
	/**
	 * For a given statement, this method returns the enclosing compound statement 
	 * @param stmt - the given statement
	 * @return - the enclosing compound statement (can be same as the compound statement in the function or may be another compound statement in it)
	 */
	private CompoundStatement getEnclosingCompoundStmt(Statement stmt){
		Traversable t = stmt;
		while( t != null && !(t instanceof CompoundStatement)){
			t = t.getParent();
		}
		return (CompoundStatement)t;
	}
	/**
	 * For a given statement, this method returns the function it is contained in 
	 * @param stmt - the given statement
	 * @return - the enclosing function
	 */
	private Procedure getEnclosingFunction(Statement stmt){
		Traversable t = stmt;
		while( t != null && !(t instanceof Procedure)){
			t = t.getParent();
		}
		return (Procedure)t;
	}
	/**
	 * returns the first non-declaration statement in the given compound statement (which may be function body)
	 * @param compoundStmt - the compound statement (which may be function body) 
	 * @return - first non-declaration statement, or null if it can't find one
	 */
	private Statement getFirstNonDeclarationStatement(CompoundStatement compoundStmt) {
		FlatIterator fi = new FlatIterator(compoundStmt);
		while(fi.hasNext()){
			Object next = fi.next();
			if(next instanceof Statement && !(next instanceof DeclarationStatement || next instanceof AnnotationStatement) )
				return (Statement)next;
		}
		return null;
	}
	/**
	 * Tells if the called function can be inlined.
	 * Currently it makes sure that the number of actual and formal parameters match and that function pointers and static variables are not used.  
	 * This method does not check for any recursion resulted from the statements inside the passed procedure
	 * That should be checked by the call graph in the calling method.
	 * @param functionToBeInlined - the function to be inlined
	 * @param fc - the function call
	 * @param enclosingFunction - the function in whose code the called function would be inlined in
	 * @return - true if the function can be inlined, false otherwise (it will later be changed to return the cost)
	 */
	private boolean canInline(Procedure functionToBeInlined, FunctionCall fc, Procedure enclosingFunction){
		// check the number of parameters
		if(fc.getNumArguments() != functionToBeInlined.getNumParameters()){
			boolean cannotInline = true;
			// handle the foo(void){} case
			if(fc.getNumArguments() == 0 && functionToBeInlined.getNumParameters() == 1){
				// List params would be null, we should get the children from the declarator, second would be the specifier 
				Declarator d = functionToBeInlined.getDeclarator();
				List c = d.getChildren();
				if(c.size() > 1){
					Object o = c.get(1);
					if(o instanceof VariableDeclaration){
						List s = ((VariableDeclaration)o).getSpecifiers();
						if(s.size() == 1 && s.get(0).equals(Specifier.VOID)) {
							cannotInline = false;
						}	
					}
				}	
			}
			if(cannotInline) {	
				if(makeLog){
					System.out.println("number of actual and formal parameters in calling " + functionToBeInlined.getName().toString() + " function in the following statement does not match so we are not inlining it");
					System.out.println(fc.toString());
				}	
				return false;
			}	
		}
		// check every parameter independently and return false in case of function pointers
		//TODO: Tools.getExpressionType() can be used to evaluate the type of expressions passed as actual parameters
		// however I'm not sure how complete this function is, so I'm not using it right now.
		// Moreover this is just an extra check as we are already assuming that the code is already compiled correctly before inlining.

		// check if the called procedure accepts function pointers, we won't inline such functions
		List params = functionToBeInlined.getParameters();
		for (int i = 0; i < params.size(); i++) {
			Declaration d = (Declaration)params.get(i);
			List<Traversable> children = d.getChildren();
			for (int j = 0; j < children.size(); j++) {
				if(children.get(j) instanceof NestedDeclarator){
					if(((NestedDeclarator)children.get(j)).isProcedure()){
						if(makeLog){
							System.out.println(functionToBeInlined.getName().toString() + " function in the following statement accepts function pointer(s) so we are not inlining it");
							System.out.println(fc.toString());
						}	
						return false;
					}	
				}
			}
		}
		// check if the called procedure has any static local variables, we won't inline such functions, unless the user has specifically
		// asked us to, in which case we would make them global variables and change their code to use new global variables
		Collection<Declaration> localVars = functionToBeInlined.getBody().getDeclarations();
		for(Declaration local : localVars){
			if(local instanceof VariableDeclaration){
				List<Specifier> specs = ((VariableDeclaration)local).getSpecifiers();
				for(Specifier spec : specs){
					if(spec.equals(Specifier.STATIC)){
						if(inlineFuncsWithStaticVars) {
							// declare the variables as globals with unique names
							TranslationUnit tUnit = getTranslationUnit(functionToBeInlined);
							Traversable t = local.getParent();
							if(t instanceof DeclarationStatement) {
								DeclarationStatement dStmt = (DeclarationStatement)t;
								CompoundStatement cStmt = functionToBeInlined.getBody();
								int n = ((VariableDeclaration)local).getNumDeclarators();
								List<Declarator> decls = new ArrayList<Declarator>();
								List<IDExpression> oldNames = new ArrayList<IDExpression>();
								List<String> newNames = new ArrayList<String>();
								for (int i = 0; i < n; i++) {
									Declarator decl = ((VariableDeclaration)local).getDeclarator(i);
									if (decl instanceof VariableDeclarator){
										decls.add(decl);
									}
								}
								for (Declarator decl : decls){
									if (decl instanceof VariableDeclarator){
										String oldName = ((VariableDeclarator)decl).getSymbolName();
										oldNames.add(((VariableDeclarator)decl).getID());
										NameID newGlobal = getUniqueIdentifier(tUnit, "", oldName);
										newNames.add(newGlobal.getName());
										((VariableDeclarator)decl).setName(newGlobal.getName());
									}
								}
								cStmt.removeStatement(dStmt);
								dStmt.detach();
								local.setParent(null);
								//local.detach();
								specs.remove(spec);
								VariableDeclaration newD = new VariableDeclaration(specs, decls);
								tUnit.addDeclarationFirst(newD);
								// also make sure that the name change gets reflected in the annotations as well
						        replaceVariableNames(functionToBeInlined.getBody(), oldNames, newNames, new HashMap<String, ArrayAccess>(), new HashMap<IDExpression, Expression>(), new LinkedList<Symbol>(), false);
							break;
							}
							else {
								System.err.println("fix me in inliner #canInine()");
							}
							
						}
						else {
							if(makeLog){
								System.out.println(functionToBeInlined.getName().toString() + " function in the following statement in function \'" + enclosingFunction.getName().toString() + "\' has static local variable(s) so we are not inlining it");
								System.out.println(fc.toString());
							}	
							return false;
						}	
					}
				}
			}	
		}
		// Make sure that the function to be inlined does not use any static external variables
		// but if the function to be inlined is in the same file as the function in whose code it is inlined then allow it
		// would be a problem if they are declared after they are used, can't extern static variables declared later.
		if(!getTranslationUnit(functionToBeInlined).equals(getTranslationUnit(enclosingFunction))){
			List<IDExpression> staticVars = getExternalStaticVars(functionToBeInlined);
			Set<Symbol> symbols_locals = functionToBeInlined.getBody().getSymbols();
			Set<Symbol> symbols_params = functionToBeInlined.getSymbols();
			Set<String> locals = new HashSet<String>(symbols_locals.size());
			Set<String> functionParams = new HashSet<String>(symbols_params.size());
			for(Symbol s : symbols_locals) {
				locals.add(s.getSymbolName());
			}
			for(Symbol s : symbols_params) {
				functionParams.add(s.getSymbolName());
			}
			
			if(staticVars.size() > 0){
				DepthFirstIterator dfi = new DepthFirstIterator(functionToBeInlined.getBody());
				Object next = null;
				HashMap<String, String> checked = new HashMap<String, String>();
				while(dfi.hasNext()){
					if((next = dfi.next()) instanceof IDExpression){
						String staticVar = ((IDExpression)next).getName();
						if(!checked.containsKey(staticVar) && !locals.contains(((IDExpression)next).toString()) && !functionParams.contains(((IDExpression)next).toString())){
							checked.put(staticVar, null);
							for (int i = 0; i < staticVars.size(); i++) {
								try{
									if(staticVars.get(i).toString().equals(staticVar)){
										if(makeLog){
											System.out.println(functionToBeInlined.getName().toString() + " function in the following statement in function \'" + enclosingFunction.getName().toString() + "\' uses external static variable(s) so we are not inlining it");
											System.out.println(fc.toString());
										}	
										return false;
									}	
								}
								catch(NullPointerException ex){
									System.err.println("Fix me in inline code..");
								}
							}
						}
					}
				}
			}
		}	
		return true;
	}
	/**
	 * returns the program the given procedure belongs to
	 * @param proc - the procedure
	 */
	private Program getProgram(Procedure proc) {
		Traversable t = proc;
		while(t != null){
			if(t instanceof Program)
				return (Program)t;
			t = t.getParent();
		}
		return null;
	}
	/**
	 * returns the translation unit the given procedure belongs to
	 * @param proc - the procedure
	 */
	private TranslationUnit getTranslationUnit(Procedure proc) {
		Traversable t = proc;
		while(t != null){
			if(t instanceof TranslationUnit)
				return (TranslationUnit)t;
			t = t.getParent();
		}
		return null;
	}
	/**
	 * Given a function, this method returns a list of static external variables found in the file of the function
	 * @param proc - the procedure of whose file we are looking into to find static external variables
	 * @return - list of static external variables
	 */
	private List<IDExpression> getExternalStaticVars(Procedure proc) {
		TranslationUnit tUnit = getTranslationUnit(proc);
		List<IDExpression> staticVars = new ArrayList<IDExpression>();
		if(tUnit != null){
			Set<Declaration> declarations = tUnit.getDeclarations();
			for(Declaration d : declarations){
				if(d instanceof VariableDeclaration){
					VariableDeclaration vd = (VariableDeclaration)d;
					List<Specifier> specs = vd.getSpecifiers();
					for (int l = 0; l < specs.size(); l++) {
						if(specs.get(l).equals(Specifier.STATIC)){
							staticVars.addAll(d.getDeclaredIDs());
						}	
					}
				}
				
			}
		}
		return staticVars;
	}
	/**
	 * For a given function this method returns a map of those global variables and their declarations that are used in 
	 * the body of the function
	 * 
	 * @param function - the function
	 * @param codeToBeInlined - body of the function in which we are trying to find the use of global variables
	 *                          note: this may be modified inlined code
	 * @param newLocals - list of new variables introduced as local variables by inlining code
	 * @param newParamNames - list of new variables introduced to replace parameters by the inlining code
	 * @return - map of global variables and their declarations
	 */
	private Map<IDExpression, Declaration> getUsedGlobalVariables(Procedure function, CompoundStatement codeToBeInlined, List<String> newLocals, List<String> newParamNames) {
		Map<IDExpression, Declaration> usedGlobalVars = new HashMap<IDExpression, Declaration> ();
		TranslationUnit tUnit = getTranslationUnit(function);
		List<SymbolTable> tables = getAllSymbolTables(tUnit);
		DepthFirstIterator dfi = new DepthFirstIterator(codeToBeInlined);
		Vector<String> functions = new Vector<String>();
		// go through the body of the function and gather all identifiers that may be global variables
		Vector<IDExpression> globalVars = new Vector<IDExpression>(); 
		Object next = null;
		while(dfi.hasNext()){
			next = dfi.next();
			if(next instanceof FunctionCall)
				functions.addElement(((FunctionCall)next).getName().toString());
			
			if(next instanceof IDExpression){
				String id = ((IDExpression)next).getName();
				if(!globalVars.contains(next) && !newLocals.contains(id) && !newParamNames.contains(id) && !functions.contains(id)){
					globalVars.addElement((IDExpression)next);
				}
			}
		}
		// for each guessed global variable, find out if it is really a global variable, if it is put it in the map to be returned 
		for (int i = 0; i < globalVars.size(); i++) {
			IDExpression id = globalVars.elementAt(i);
			for (int j = 0; j < tables.size(); j++) {
				Set<Symbol> symbols = tables.get(j).getSymbols();
				Set<Declaration> declarations = tables.get(j).getDeclarations();
				for (Symbol s : symbols) {
					if(s.getSymbolName().equals(id.getName())) {
						for (Declaration d : declarations) {
							if(d.equals(s.getDeclaration()))
								usedGlobalVars.put(id, d);
						}	
					}	
				}
			}
		}
		return usedGlobalVars;
	}
	/**
	 * For the given translation unit, this method returns all symbol tables (including the given, as well as the parent)
	 */
	private List<SymbolTable> getAllSymbolTables(TranslationUnit tUnit) {
		List<SymbolTable> tables = new ArrayList<SymbolTable>();
		Traversable t = tUnit;
		while(t != null){
			if(t instanceof SymbolTable)
				tables.add((SymbolTable)t);
			
			t = t.getParent();
		}
		return tables;
	}
	/**
	 * A variable declaration can have many declarators, this method returns the one which involves the given identifier
	 * @param declaration - the variable declaration
	 * @param id - the identifier we are interested in
	 * @return - the declarator involving the given identifier, null if there is none
	 */
	private Declarator getDeclarator(VariableDeclaration declaration, IDExpression id){
		int n = declaration.getNumDeclarators();
		for (int i = 0; i < n; i++) {
			Declarator decl = declaration.getDeclarator(i);
			Declarator d = null;
			if(decl instanceof VariableDeclarator) {
				d = (VariableDeclarator)decl;
			}	
			else if(decl instanceof NestedDeclarator){
				Declarator decl2 = (NestedDeclarator)decl; 
				while(decl2 instanceof NestedDeclarator){
					Declarator tempdecl = decl2;
					decl2 = ((NestedDeclarator)decl2).getDeclarator();
					if(decl2 == tempdecl)
						break;
				}
				d = decl2;
			}
			else {
				System.err.println("fix me in inliner #getDeclarator()");
			}
			if(d instanceof VariableDeclarator) {
				if(id.getName().equals(((VariableDeclarator)d).getSymbolName())) // but usually direct declarator is empty so we need to find the id in children
					return decl;
				List<Traversable> children = d.getChildren();
				for (int j = 0; j < children.size(); j++) {
					if(children.get(j) instanceof IDExpression && id.getName().equals(((IDExpression)children.get(j)).getName())){
						return decl;
					}
				}
			}	
		}
		return null;
	}
	/**
	 * tells if the body of the function supplied, has multiple return statements or not
	 * @param functionBody - compound statement representing function body
	 * @return - true if the function body has multiple return statements in it, false if it has one or zero
	 */
	private boolean hasMultipleReturnStmts(CompoundStatement functionBody){
        DepthFirstIterator dfi = new DepthFirstIterator(functionBody);
        boolean returnStatement = false;
        while(dfi.hasNext()){
        	Object obj = dfi.next();
        	// if there is a return statement, there should be no statement after it.
        	if(returnStatement) {
        	   	if(obj instanceof Statement && !(obj instanceof NullStatement))
        	   		return true;
        	}
        	if(obj instanceof ReturnStatement){
        		if(returnStatement)
        			return true;
        		returnStatement = true;
        	}
        }
        return false;
	}
	/**
	 * returns a list containing function declarations and definitions visible in the specified translation unit
	 * @param tUnit - the translation unit
	 * @return - list of visible function declarations and definitions
	 */
	private List getAvailableFunctions(TranslationUnit tUnit){
		List functions = new LinkedList();
		FlatIterator fi = new FlatIterator(tUnit);
		Object obj = null;
		while(fi.hasNext()){
			obj = fi.next();
			
			if(obj instanceof VariableDeclaration){
				VariableDeclaration d = (VariableDeclaration)obj;
				for(int i = 0; i < d.getNumDeclarators(); i++){
					if(d.getDeclarator(i) instanceof ProcedureDeclarator){
						functions.add(d);
						break;
					}
				}
			}
			if(obj instanceof Procedure){
				functions.add(obj);
			}
			
		}
		return functions;
	}
	/**
	 * tells if the function declaration/definition of the function called in the specified function call is available in 
	 * the provided translation unit
	 * @param fc - the function call
	 * @param tUnit - the translation unit
	 * @return - true if the declaration/definition of the function is available/visible, false otherwise
	 */
	private boolean isDeclarationAvailable(FunctionCall fc, TranslationUnit tUnit){
		return getFunctionDeclaration(fc, tUnit) != null;
	}
	
	/**
	 * returns the declaration/definition of the function, called in the passed function call, if it is available in the 
	 * provided translation unit, null otherwise
	 * @param fc - the function call
	 * @param tUnit - the translation unit
	 * @return - declaration/definition of the function if it is available in the passed translation unit, null otherwise
	 */
	private Object getFunctionDeclaration(FunctionCall fc, TranslationUnit tUnit){
		String name = fc.getName().toString();
		// check if we have the function or its declaration
		List availableFuncs = getAvailableFunctions(tUnit);
		boolean declared = false;
		for (int k = 0; k < availableFuncs.size(); k++) {
			Object o = availableFuncs.get(k);
			if(o instanceof Procedure){
				// NOTE: we are only comparing function names (good enough for C, as no function overriding)
				if(((Procedure)o).getSymbolName().equals(name)){
					return o;
				}
			}
			else if(o instanceof VariableDeclaration){
				VariableDeclaration v = (VariableDeclaration)o;
				for (int l = 0; l < v.getNumDeclarators(); l++) {
					if(v.getDeclarator(l) instanceof ProcedureDeclarator){
						if(((ProcedureDeclarator)v.getDeclarator(l)).getSymbolName().equals(name)){
							return v;
						}
					}
				}
			}
		}
		return null;
	}
//	/**
//	 * returns a list containing all typedefs for the given translation unit
//	 * @param tUnit - the translation unit
//	 * @return - list containing all typedefs
//	 */
//	private List<VariableDeclaration> getTypedefs(TranslationUnit tUnit){
//		List<VariableDeclaration> typedefs = new LinkedList<VariableDeclaration>();
//		FlatIterator fi = new FlatIterator(tUnit);
//		Object obj = null;
//		while(fi.hasNext()){
//			obj = fi.next();
//			
//			if(obj instanceof VariableDeclaration){
//				VariableDeclaration d = (VariableDeclaration)obj;
//				List<Specifier> specs = d.getSpecifiers();
//				for(int i = 0; i < specs.size(); i++){
//					if(Specifier.TYPEDEF.equals(specs.get(i))){
//						typedefs.add(d);
//						break;
//					}
//				}
//			}
//		}
//		return typedefs;
//	}
	/**
	 * Given a list of InlineAnnotation instances, this method returns the "inlinein" or "noinlinein" annotation. 
	 * It returns null if both are present and prints a warning message on the console
	 */
	private InlineAnnotation getInlineInAnnotation(List<InlineAnnotation> annotations){
		InlineAnnotation annot = null;
		for (Iterator<InlineAnnotation> iterator = annotations.iterator(); iterator.hasNext();) {
			InlineAnnotation inlineAnnotation = iterator.next();
			if(inlineAnnotation.equals(InlineAnnotation.INLINE_IN) || inlineAnnotation.equals(InlineAnnotation.NO_INLINE_IN)) {
				if(annot == null) {
					annot = inlineAnnotation;
				}
				else {
					System.out.println("Warning: InlineExpansion: Found conflicting inline pragmas, going to ignoring them");
					return null;
				}
			}
		}
		return annot;
	}

	/**
	 * Given a list of InlineAnnotation instances, this method returns the "inline" or "noinline" annotation. 
	 * It returns null if both are present and prints a warning message on the console
	 */
	private InlineAnnotation getInlineAnnotation(List<InlineAnnotation> annotations){
		InlineAnnotation annot = null;
		for (Iterator<InlineAnnotation> iterator = annotations.iterator(); iterator.hasNext();) {
			InlineAnnotation inlineAnnotation = iterator.next();
			if(inlineAnnotation.equals(InlineAnnotation.INLINE) || inlineAnnotation.equals(InlineAnnotation.NO_INLINE)) {
				if(annot == null) {
					annot = inlineAnnotation;
				}
				else {
					System.out.println("Warning: InlineExpansion: Found conflicting inline pragmas, going to ignoring them");
					return null;
				}
			}
		}
		return annot;
	}
	/**
	 * Given a list of procedures this function returns the topological list of these and all functions called from these functions
	 * @param callGraph - the call graph
	 * @param procs - list of functions
	 * @param except - list of functions which we won't inilin in
	 */
	private ArrayList<Procedure> getFunctionsRecursively(CallGraph callGraph, ArrayList<Procedure> procs, ArrayList<Procedure> except) {
		Set<Procedure> unvisited = new HashSet<Procedure>();
		unvisited.addAll(callGraph.getCallGraph().keySet());

		ArrayList<Procedure> sorted_list = new ArrayList<Procedure>(unvisited.size());
		for (Procedure proc : procs) {
			unvisited.remove(proc);
			topologicalSort(callGraph, proc, unvisited, sorted_list, except);
		}	
		return sorted_list;
	}
	/**
	 * performs the topological sort recursively
	 * @param callGraph - the call graph
	 * @param proc - the procedure whose callees have to be added recursively
	 * @param unvisited - list of unvisited functions
	 * @param sorted_list - list of visited functions
	 * @param except - list of functions which we won't inilin in
	 */
	private void topologicalSort(CallGraph callGraph, Procedure proc, Set<Procedure> unvisited, List<Procedure> sorted_list, ArrayList<Procedure> except)
	{
		Node node = (Node)callGraph.getCallGraph().get(proc);
		unvisited.remove(proc);
		if(!except.contains(proc)) {
			for ( Procedure callee : (ArrayList<Procedure>)node.getCallees() )
			{
				if (unvisited.contains(callee))
				{
					topologicalSort(callGraph, callee, unvisited, sorted_list, except);
				}
			}
			sorted_list.add(proc);
		}	
	}
}
