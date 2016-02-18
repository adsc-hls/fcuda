package cetus.analysis;

import cetus.hir.*;
import java.util.*;

/**
 * Banerjee Test implements data-dependence testing for a pair of
 * affine subscripts using Banerjee inequalities
 */
public class BanerjeeTest implements DDTest {

    long getPositivePart(long a) {
        if (a >= 0) {
            return a;
        } else {
            return 0;
        }
    }

    long getNegativePart(long a) {
        if (a <= 0) {
            return a;
        } else {
            return 0;
        }
    }

    static final int LB = 0;
    static final int UB = 4;

    static final int LB_any = 0;
    static final int LB_less = 1;
    static final int LB_equal = 2;
    static final int LB_greater = 3;

    static final int UB_any = 4;
    static final int UB_less = 5;
    static final int UB_equal = 6;
    static final int UB_greater = 7;

    /**
     * Stores bounds calculated as per banerjee's inequalities for every loop
     * in the nest
     */
    HashMap<Loop, Vector<Long>> banerjee_bounds;

    /** Subscript pair related information required by Banerjee test */
    private Expression subscript1, subscript2;

    /** Loop nest information */
    private LinkedList<Loop> loopnest;

    /** Loop information of the relevant loops */
    private HashMap<Loop, LoopInfo> loopInfo;
    
    /**
     * Constant coefficients in the affine subscript expressions as identified
     * by normalising the expression
     */
    Expression const1, const2;

    /** Eligibility of the test */
    boolean eligible_for_test;

    /**
     * Constructs a new test problem with the specified {@code SubscriptPair}.
     *
     * @param pair the input {@code SubscriptPair} object.
     */
    public BanerjeeTest (SubscriptPair pair) {
        this.subscript1 = pair.getSubscript1();
        this.subscript2 = pair.getSubscript2(); 
        this.loopnest = pair.getEnclosingLoopsList(); 
        this.loopInfo = pair.getEnclosingLoopsInfo();
        
        banerjee_bounds = new HashMap<Loop,Vector<Long>>();
        long A = 0, B = 0, C = 0;
        List<Identifier> idlist = new ArrayList<Identifier>();
        for (Loop loop_id : loopnest) {
            LoopInfo info = loopInfo.get(loop_id);
            Identifier index = (Identifier)info.getLoopIndex();
            idlist.add(index);
        }

        this.const1 = Symbolic.getConstantCoefficient(subscript1, idlist);
        this.const2 = Symbolic.getConstantCoefficient(subscript2, idlist);

        //For each LoopInfo object, compute the Banerjee bounds 
        //and add it to the map
        for (Loop loop: loopnest) {
            LoopInfo info = loopInfo.get(loop);
            Identifier id = (Identifier)info.getLoopIndex();
            PrintTools.println("indexVariable " + id.getName(), 2);
            Vector<Long> bounds = new Vector<Long> (8); //Banerjee bounds

            // Check for loop invariance of constant term in subscript
            if (LoopTools.isLoopInvariant(loop, this.const1) &&
                    LoopTools.isLoopInvariant(loop, this.const2)) {
                this.eligible_for_test = true;
            } else {
                this.eligible_for_test = false;
                break;
            }
            // Check if coefficients of index variables are constant integer
            // values
            Expression c1 = Symbolic.getCoefficient(subscript1, id);
            Expression c2 = Symbolic.getCoefficient(subscript2, id);
            if ((c1 instanceof IntegerLiteral) &&
                    (c2 instanceof IntegerLiteral)) {
                A = ((IntegerLiteral)c1).getValue();
                B = ((IntegerLiteral)c2).getValue();
                this.eligible_for_test = true;
            } else {
                this.eligible_for_test = false;
                break;
            }

            // Check if loop bounds and increment are compatible with testing
            long U, L, N;
            Expression u = info.getLoopUB();
            Expression l = info.getLoopLB();
            Expression n = info.getLoopIncrement();
            if ((u instanceof IntegerLiteral) &&
                    (l instanceof IntegerLiteral) &&
                    (n instanceof IntegerLiteral)) {
                U = ((IntegerLiteral)u).getValue();
                L = ((IntegerLiteral)l).getValue();
                N = ((IntegerLiteral)n).getValue();
                PrintTools.println("upperBound " + U, 2);
                PrintTools.println("lowerBound " + L, 2);
                PrintTools.println("lowerBound " + L, 2);
                this.eligible_for_test = true;
            } else {
                this.eligible_for_test = false;
                break;
            }

            if (N >= 0) {
                bounds.add(BanerjeeTest.LB_any, new Long(
                        (getNegativePart(A)-getPositivePart(B))*(U-L) +
                        (A-B)*L));
                bounds.add(BanerjeeTest.LB_less, new Long(
                        getNegativePart(getNegativePart(A)-B)*(U-L-N) +
                        (A-B)*L - B*N));
                bounds.add(BanerjeeTest.LB_equal, new Long(
                        getNegativePart(A-B)*(U-L) + (A-B)*L));
                bounds.add(BanerjeeTest.LB_greater, new Long(
                        getNegativePart(A-getPositivePart(B))*(U-L-N) +
                        (A-B)*L + A*N));
                bounds.add(BanerjeeTest.UB_any, new Long(
                        (getPositivePart(A)-getNegativePart(B))*(U-L) +
                        (A-B)*L));
                bounds.add(BanerjeeTest.UB_less, new Long(
                        getPositivePart(getPositivePart(A)-B)*(U-L-N) +
                        (A-B)*L - B*N));
                bounds.add(BanerjeeTest.UB_equal, new Long(
                        getPositivePart(A-B)*(U-L) + (A-B)*L));
                bounds.add(BanerjeeTest.UB_greater, new Long(
                        getPositivePart(A-getNegativePart(B))*(U-L-N) +
                        (A-B)*L + A*N));
            } else { // Negative stride
                // Switch loop bounds
                long temp_bound = U;
                U = L;
                L = temp_bound;
                // Use the absolute value of the stride
                N = ((-1)*N);
                bounds.add(BanerjeeTest.LB_any, new Long(
                        (getNegativePart(A)-getPositivePart(B))*(U-L) +
                        (A-B)*L));
                bounds.add(BanerjeeTest.LB_less, new Long(
                        getNegativePart(A-getPositivePart(B))*(U-L-N) +
                        (A-B)*L + A*N));
                bounds.add(BanerjeeTest.LB_equal, new Long(
                        getNegativePart(A-B)*(U-L) + (A-B)*L));
                bounds.add(BanerjeeTest.LB_greater, new Long(
                        getNegativePart(getNegativePart(A)-B)*(U-L-N) +
                        (A-B)*L - B*N));
                bounds.add(BanerjeeTest.UB_any, new Long(
                        (getPositivePart(A)-getNegativePart(B))*(U-L) +
                        (A-B)*L));
                bounds.add(BanerjeeTest.UB_less, new Long(
                        getPositivePart(A-getNegativePart(B))*(U-L-N) +
                        (A-B)*L + A*N));
                bounds.add(BanerjeeTest.UB_equal, new Long(
                        getPositivePart(A-B)*(U-L) + (A-B)*L));
                bounds.add(BanerjeeTest.UB_greater, new Long(
                        getPositivePart(getPositivePart(A)-B)*(U-L-N) +
                        (A-B)*L - B*N));
            }
            banerjee_bounds.put(loop, bounds);
        }        
    }

    /*
     * (non-Javadoc)
     * @see cetus.analysis.DDTest#isTestEligible()
     */
    public boolean isTestEligible() {
        return this.eligible_for_test;
    }
    
    /*
     * (non-Javadoc)
     * @see cetus.analysis.DDTest#testDependence(cetus.analysis.SubscriptPair,
     * cetus.analysis.DependenceVector)
     */
    public boolean testDependence(DependenceVector dependence_vector) {
        long banerjeeLB=0;
        long banerjeeUB=0;
        long diff = 0;
        Expression expr_diff = Symbolic.subtract(const2, const1);
        if (expr_diff instanceof IntegerLiteral) {
            diff = ((IntegerLiteral)expr_diff).getValue();
        } else {
            // Difference in constants is a symbolic expression
            // Conservatively mark dependence for this subscript pair
            return true;
        }
        LinkedList<Loop> nest = this.loopnest;
        for (Loop loop : nest) {
            int loop_dependence_direction =
                    dependence_vector.getDirection(loop);
            banerjeeLB += banerjee_bounds.get(loop).get(
                    loop_dependence_direction+BanerjeeTest.LB);
            banerjeeUB += banerjee_bounds.get(loop).get(
                    loop_dependence_direction+BanerjeeTest.UB);
        }
        if (diff < banerjeeLB || diff > banerjeeUB) {
            PrintTools.println("Dependence does not exist", 2);
            printDirectionVector((dependence_vector.getDirectionVector()),nest);
            return false;
        } else {
            PrintTools.println("Dependence exists", 2);
            printDirectionVector((dependence_vector.getDirectionVector()),nest);
            return true;
        }
    }

    private void printDirectionVector(
            HashMap<Loop,Integer> dv, LinkedList<Loop> nest) {
        PrintTools.print("(", 2);
        for (int i=0; i< nest.size(); i++) {
            Loop loop = nest.get(i);
            PrintTools.print(DependenceVector.depstr[dv.get(loop)], 2);
        }
        PrintTools.println(")", 2);
    }
    
    public LinkedList<Loop> getCommonEnclosingLoops() {
        return this.loopnest;
    }
}
