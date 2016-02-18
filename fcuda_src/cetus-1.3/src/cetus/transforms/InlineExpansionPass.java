package cetus.transforms;

import java.util.ArrayList;
import java.util.StringTokenizer;

import cetus.analysis.InlineExpansion;
import cetus.exec.Driver;
import cetus.hir.Program;

/**
 * Transforms a program by performing simple subroutine in-line expansion in its main function.
 */

public class InlineExpansionPass extends TransformPass {
	
	/** Name of the inline expansion pass */
	private static final String NAME = "[InlineExpansionPass]";
	private static final String MODE = "mode";
	private static final String DEPTH = "depth";
	private static final String PRAGMA = "pragma";
	private static final String DEBUG = "debug";
	private static final String FORONLY = "foronly";
	private static final String FUNCTIONS = "functions";
	private static final String COMPLEMENT = "complement";
	
	/**
	 * Constructs an inline expansion pass 
	 * @param program - the program to perform inline expansion on
	 */
	public InlineExpansionPass(Program program) {
		super(program);
	}
	
	@Override
	public void start() {
		String options = Driver.getOptionValue("tinline");
		InlineExpansion inlineExpansion = new InlineExpansion();
		StringTokenizer tokenizer = new StringTokenizer(options, ":");
		String option;
		while(tokenizer.hasMoreTokens()) {
			option = tokenizer.nextToken();
			int eqIndex = option.indexOf('='); 
			if( eqIndex != -1) {
				String opt = option.substring(0, eqIndex).trim();
				if(!opt.equals(FUNCTIONS)) {
					try {
						int value = new Integer(option.substring(eqIndex+1).trim()).intValue();
						if(opt.equals(MODE)) {
							inlineExpansion.setMode(value);
						}
						else if(opt.equals(DEBUG)) {
							inlineExpansion.setDebugOption(value == 1? true : false);
						}
						else if(opt.equals(PRAGMA)) {
							inlineExpansion.setHonorPragmas(value == 1? true : false);
						}
						else if(opt.equals(DEPTH)) {
							inlineExpansion.setLevel_1(value == 1? true : false);
						}
						else if(opt.equals(FORONLY)) {
							inlineExpansion.setInsideForOnly(value == 1? true : false);
						}
						else if(opt.equals(COMPLEMENT)) {
							inlineExpansion.setComplementFunctions(value == 1? true : false);
						}
					}
					catch(NumberFormatException ex){
					}
				}
				else {
					StringTokenizer funcs = new StringTokenizer(option.substring(eqIndex+1), " ,");
					ArrayList<String> functions = new ArrayList<String>();
					while(funcs.hasMoreTokens()) {
						functions.add(funcs.nextToken().trim());
					}
					inlineExpansion.setCommandlineFunctions(functions);
				}
			}
		}
		
		inlineExpansion.inline(program);
	}
	
	@Override
	public String getPassName() {
		return NAME;
	}
}
