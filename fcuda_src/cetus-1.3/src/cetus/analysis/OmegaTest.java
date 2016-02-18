package cetus.analysis;

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Loop;
import cetus.hir.PrintTools;

import java.util.HashMap;
import java.util.LinkedList;


public class OmegaTest implements DDTest 
{
	/* Private member storage for access by native methods */
	Expression a1, a2;
	LinkedList<Loop> enclosing_nest;
	HashMap<Loop, LoopInfo> loop_information;

	/* Data required for Omega Test Interface creation */
	long[] common_loop_steps;
	int loop_nest_size;
	
	/* Constant coefficients in the affine subscript expressions as identified
	 * by normalising the expression */	
	Expression const1, const2;
	boolean eligible_for_test;

	public OmegaTest (DDArrayAccessInfo a1,
						DDArrayAccessInfo a2,
						LinkedList<Loop> common_enclosing_loops,
						HashMap<Loop, LoopInfo> loopInfo)
	{
		long A = 0, B = 0, C = 0;
		long U = 0, L = 0, N = 0;
		
		this.enclosing_nest = common_enclosing_loops; 
		this.loop_information = loopInfo;

		/* Build data structures to store information that will be needed by the Omega test
		 * interface */
		this.common_loop_steps = new long[enclosing_nest.size()];
		int loop_id = 0;
		for (Loop loop: enclosing_nest)
		{
				LoopInfo info = loop_information.get(loop);
				Expression n = info.getLoopIncrement();
				if (n instanceof IntegerLiteral)
					N = ((IntegerLiteral)n).getValue();
				else
					N = 0;
				this.common_loop_steps[loop_id] = N;
				loop_id++;
		}
		this.loop_nest_size = enclosing_nest.size();
		
		/* JNI Interface to Omega Library, see ./omega_lib/omega_test_interf.c */
		omega_interf_initialize();		
	}
	
	/* JNI Interface functions for omega test */
	private native boolean omega_test();
	private native void omega_interf_initialize();
	static {
		System.loadLibrary("Omega");
	}
	
	/*
	 * (non-Javadoc)
	 * @see cetus.analysis.DDTest#isTestEligible()
	 */
	public boolean isTestEligible()
	{
		return this.eligible_for_test;
	}
	
	/*
	 * (non-Javadoc)
	 * @see cetus.analysis.DDTest#testDependence(cetus.analysis.SubscriptPair, cetus.analysis.DependenceVector)
	 */
	public boolean testDependence(DependenceVector dependence_vector)
	{
		return true;
	}
	
	public void printDirectionVector(HashMap<Loop,Integer> dv, LinkedList<Loop> nest)
	{
		PrintTools.print("(", 2);
		for (int i=0; i< nest.size(); i++)
		{
			Loop loop = nest.get(i);
			PrintTools.print(DependenceVector.depstr[dv.get(loop)], 2);
		}
		PrintTools.println(")", 2);
	}
	
	public LinkedList<Loop> getCommonEnclosingLoops()
	{
		return this.enclosing_nest;
	}
}

