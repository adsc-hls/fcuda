package cetus.analysis;

import cetus.exec.Driver;
import cetus.hir.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

/**
 * Wrapper framework for executing specific data-dependence test on array
 * subscripts
 */
public class DDTestWrapper {
    // Store the array accesses for which this wrapper will perform dependence
    // testing
    private DDArrayAccessInfo acc1, acc2;
    // Common eligible nest for the two accesses which will provide the
    // dependence vector
    private LinkedList<Loop> common_eligible_nest;
    // Loop Info for all loops pertaining to these two accesses
    private HashMap<Loop, LoopInfo> loopinfo;
    // Multiple dependence tests can be used for testing
    private static final int DDTEST_BANERJEE = 1;
    private static final int DDTEST_RANGE = 2;
    private static final int DDTEST_OMEGA = 3;
    // Get Commandline input for which test must be run,
    // default = DDTEST_BANERJEE
    private int ddtest_type;

    /**
    * Constructs a new test wrapper with the specified pair of array accesses
    * and loop information.
    */
    public DDTestWrapper(DDArrayAccessInfo a1,
                         DDArrayAccessInfo a2,
                         LinkedList<Loop> common_eligible_nest,
                         HashMap<Loop, LoopInfo> loopInfo) {
        // Array accesses and their information required for testing
        this.acc1 = a1;
        this.acc2 = a2;
        this.common_eligible_nest = common_eligible_nest;
        this.loopinfo = loopInfo;
        this.ddtest_type =
                Integer.valueOf(Driver.getOptionValue("ddt")).intValue();
    }
    
    private int getDDTestType() {
        return ddtest_type;
    }

    private DDArrayAccessInfo getAcc1() {
        return acc1;
    }

    private DDArrayAccessInfo getAcc2() {
        return acc2;
    }

    private LinkedList<Loop> getCommonEnclosingLoops() {
        return common_eligible_nest;
    }

    private HashMap<Loop, LoopInfo> getAllLoopsInfo() {
        return loopinfo;
    }

    /**
     * Accepts two access pairs, partitions their subscripts, performs
     * dependence testing and constructs a set of dependence vectors if
     * dependence exists
     * @param DVset
     * @return true if dependence exists
     */
    public boolean testAccessPair(ArrayList<DependenceVector> DVset) {
        boolean dependence_result = true;
        int ddtest_type = getDDTestType();
        if (ddtest_type == DDTEST_OMEGA) {
            dependence_result = testAllSubscriptsTogether(DVset);
        }
        // Add other dependence tests here when required
        // else if ...
        // By default, use Banerjee DDTEST_BANERJEE = 1
        else if (ddtest_type == DDTEST_BANERJEE ||
                 ddtest_type == DDTEST_RANGE) {
            dependence_result = testSubscriptBySubscript(DVset);
        }
        return dependence_result;
    }

    private boolean
            testAllSubscriptsTogether(ArrayList<DependenceVector> DVset) {
        DDTest ddtest = null;
        ArrayList<DependenceVector> returned_DVset;
        int ddtest_type = getDDTestType();
        if (ddtest_type == DDTEST_OMEGA) {
            // Currently, the Omega Test inclusion in Cetus is on hold. This 
            // section of code is not meant to be used by Cetus users. 
            // Remove the warning and exit call once Omega test inclusion 
            // is complete 
            // ddtest = new OmegaTest(getAcc1(),
            //                        getAcc2(),
            //                        getCommonEnclosingLoops(),
            //                        getAllLoopsInfo());
            throw new UnsupportedOperationException(
                    "OMEGA test is not supported now");
        }
        // else if .. add other whole array access tests here
        returned_DVset = testAllDependenceVectors(ddtest);
        if (returned_DVset.size() == 0) {
            return false;
        } else {
            // Merge returned set with input DVset
            mergeVectorSets(DVset, returned_DVset);
            return true;
        }
    }

    private boolean
            testSubscriptBySubscript(ArrayList<DependenceVector> DVset) {
        ArrayList<SubscriptPair> subscriptPairs;
        LinkedList<HashSet<SubscriptPair>> partitions;
        ArrayAccess access1 = getAcc1().getArrayAccess();  //(ArrayAccess)acc1;
        ArrayAccess access2 = getAcc2().getArrayAccess();  //(ArrayAccess)acc2;
        Statement stmt1 = getAcc1().getParentStatement();
        Statement stmt2 = getAcc2().getParentStatement();
        // test dependency only if two array accesses with the same array
        // symbol id and have the same dimension
        if (access1.getNumIndices() == access2.getNumIndices()) {
            // Obtain single subscript pairs and while doing so, check if the
            // subscripts are affine
            int dimensions = access1.getNumIndices();
            subscriptPairs = new ArrayList<SubscriptPair>(dimensions);
            for (int dim = 0; dim < dimensions; dim++) {
                SubscriptPair pair = new SubscriptPair(
                        access1.getIndex(dim),
                        access2.getIndex(dim),
                        stmt1,
                        stmt2,
                        getCommonEnclosingLoops(),
                        getAllLoopsInfo());
                subscriptPairs.add(dim, pair);
            }
            // Partition the subscript pairs - currently ignore effects of
            // coupled subscripts
            partitions = getSubscriptPartitions(subscriptPairs);
            for (HashSet<SubscriptPair> partition : partitions) {
                //only singletons -> currently it is always one (smin)
                if (partition.size() == 1) {
                    boolean depExists =
                            testSeparableSubscripts(partition, DVset);
                    if (!depExists) {
                        return false;
                    }
                } else {
                    // Error condition
                    PrintTools.println("testAccessPair: partition.size()=" +
                            partition.size(), 0);
                    // Partition size in error, conservatively return true
                    SubscriptPair pair = partition.iterator().next();
                    DependenceVector dv =
                            new DependenceVector(pair.getEnclosingLoopsList());
                    if (!DVset.contains(dv)) {
                        DVset.add(dv);
                    }
                }
            }
        } else {
            // For arrays with different dimensions that are said to be aliased,
            // conservatively assume dependence in all directions with respect
            // to enclosing loops
            DependenceVector dv =
                    new DependenceVector(getCommonEnclosingLoops());
            if (!DVset.contains(dv)) {
                DVset.add(dv);
            }
        }
        // Dependence exists
        return true;
    }

    // Caution: call this only after all subscriptPairs are found
    private LinkedList<HashSet<SubscriptPair>>
            getSubscriptPartitions(ArrayList<SubscriptPair> subscriptPairs) {
        // for now they are all separable
        LinkedList<HashSet<SubscriptPair>> partitions =
                new LinkedList<HashSet<SubscriptPair>>();
        // this may look redundant now, but all the partitions are singletons 
        // containing a SubscriptPair, in the future, a more elaborate 
        // partition algorithm will be incorporated along with a coupled
        // subscript test
        PrintTools.println("getSubscriptPartitions: subscriptPairs.size()=" +
                subscriptPairs.size(), 2);
        for (int i = 0; i < subscriptPairs.size(); i++) {
            SubscriptPair pair = subscriptPairs.get(i);
            HashSet<SubscriptPair> new_partition = new HashSet<SubscriptPair>();
            new_partition.add(pair);
            // In order to test simpler ZIV subscripts first, we add them
            // to the beginning of the partition list
            // ------------------------------------------------------
            if (pair.getComplexity() == 0) {
                partitions.addFirst(new_partition);
            } else {
                partitions.addLast(new_partition);
            }
            // ------------------------------------------------------
        }
        return partitions;
    }

    private boolean testSeparableSubscripts(HashSet<SubscriptPair> partition,
                                            ArrayList<DependenceVector> DVset) {
        boolean depExists;
        // iterate over partitions and get singletons
        ArrayList<DependenceVector> DV = new ArrayList<DependenceVector>();
        // get the first (AND ONLY) element
        SubscriptPair pair = partition.iterator().next();
        switch (pair.getComplexity()) {
        case 0:
            PrintTools.println("** calling testZIV", 2);
            depExists = testZIV(pair, DV);
            break;
        case 1:
        default:
            PrintTools.println("** calling testMIV: Complexity=" +
                    pair.getComplexity(), 2);
            depExists = testMIV(pair, DV);
            break;
        }
        if (!depExists) {
            return depExists;
        } else {
            this.mergeVectorSets(DVset, DV);
            return true;
        }
    }

    /**
     * For each vector in DVSet, replicate it ||DV|| times, merge each replica
     * with one vector in DV (DV is the set of vectors returned by the
     * dependence test) Add the new merged vector back to DVSet only if it is a
     * valid vector
     */
    private void mergeVectorSets(ArrayList<DependenceVector> DVset,
                                 ArrayList<DependenceVector> DV) {
        if (DVset.size() > 0) {
            ArrayList<DependenceVector> auxDVset =
                    new ArrayList<DependenceVector>();
            auxDVset.addAll(DVset);
            DVset.removeAll(auxDVset);
            for (int i = 0; i < auxDVset.size(); i++) {
                DependenceVector dv = auxDVset.get(i);
                for (int j = 0; j < DV.size(); j++) {
                    DependenceVector dv_aug = DV.get(j);
                    DependenceVector new_dv = new DependenceVector(dv);
                    new_dv.mergeWith(dv_aug);
                    // Add the merged dependence vector only if it contains
                    // valid directions
                    if (new_dv.isValid()) {
                        DVset.add(new_dv);
                    }
                }
            }
        } else {
            DVset.addAll(DV);
        }
        return;
    }

    private boolean testZIV(SubscriptPair pair,
                            ArrayList<DependenceVector> DV) {
        Expression subscript1 = pair.getSubscript1();
        Expression subscript2 = pair.getSubscript2();
        Expression expr_diff = Symbolic.subtract(subscript1, subscript2);
        if (expr_diff instanceof IntegerLiteral) {
            IntegerLiteral diff = (IntegerLiteral)expr_diff;
            if (diff.getValue() == 0) {
                // Need to assign all possible combinations of DVs to this
                // subscript pair
                DependenceVector dv =
                        new DependenceVector(pair.getEnclosingLoopsList());
                //for (Loop l : pair.getEnclosingLoopsList()) {
                //  dv.setDirection(l, DependenceVector.equal);
                //}
                DV.add(dv);
                return true;
            } else {
                return false;
            }
        } else {
            // Difference in expressions is symbolic, conservatively return true
            DependenceVector dv =
                    new DependenceVector(pair.getEnclosingLoopsList());
            DV.add(dv);
            return true;
        }
    }

    /**
     * Having collected all information related to subscripts and enclosing
     * loops, this is the function that will call the dependence test for MIV
     * (and currently SIV) subscripts.
     */
    private boolean testMIV(SubscriptPair pair,
                            ArrayList<DependenceVector> dependence_vectors) {
        DDTest ddtest = null;
        ArrayList<DependenceVector> new_dv;
        int ddtest_type = getDDTestType();
        if (ddtest_type == DDTEST_OMEGA) {
            // ERROR, how did we get here?
            PrintTools.println("Error in data dependence testing", 0);
            Tools.exit(0);
        }
        // Add other subscript by subscript dependence tests here when required
        // else if ...
        // By default, use Banerjee
        else if (ddtest_type == DDTEST_RANGE) {
            ddtest = RangeTest.getInstance(pair);
        } else if (ddtest_type == DDTEST_BANERJEE) {
            ddtest = new BanerjeeTest(pair);
        }
        if (ddtest.isTestEligible()) {
            new_dv = testAllDependenceVectors(ddtest);
            if (new_dv.size() == 0) {
                return false;
            } else {
                dependence_vectors.addAll(new_dv);
                return true;
            }
        } else {
            DependenceVector dv =
                    new DependenceVector(pair.getEnclosingLoopsList());
            dependence_vectors.add(dv);
            return true;
        }
    }

    /** 
     * Test all combinations of dependence vectors for the enclosing loop nest,
     * prune on direction vectors for which no dependence exists.
     */
    private ArrayList<DependenceVector>
            testAllDependenceVectors(DDTest ddtest) {
        ArrayList<DependenceVector> dv_list = new ArrayList<DependenceVector>();
        LinkedList<Loop> nest = ddtest.getCommonEnclosingLoops();
        //create vector dv=(*,...,*);
        DependenceVector dv = new DependenceVector(nest); 
        // test dependence vector tree starting at (*,*,*,....) vector
        if (ddtest.testDependence(dv)) {
            // Test entire tree only if dependence exists in the any(*)
            // direction
            testTree(ddtest, dv, 0, dv_list);
        }
        return dv_list;
    }

    private void testTree(DDTest ddtest,
                          DependenceVector dv,
                          int pos,
                          ArrayList<DependenceVector> dv_list) {
        LinkedList<Loop> nest = ddtest.getCommonEnclosingLoops();
        // Test the entire tree of dependence vectors, prune if dependence
        // doesn't exist at a given level i.e. don't explore the tree further
        for (int dir = DependenceVector.less;
                dir <= DependenceVector.greater; dir++) {
            Loop loop = nest.get(pos);
            dv.setDirection(loop, dir);
            if (ddtest.testDependence(dv)) {
                DependenceVector dv_clone = new DependenceVector(dv);
                // Add to dependence vector list only if it does not contain
                // the 'any' (*) direction for all given loops
                if (!((dv_clone.getDirectionVector()).
                        containsValue(DependenceVector.any))) {
                    dv_list.add(dv_clone);
                }
                // Dependence exists, hence test the child tree rooted at
                // current dv
                if ((pos + 1) < nest.size()) {
                    testTree(ddtest, dv, pos + 1, dv_list);
                }
            }
            dv.setDirection(loop, DependenceVector.any);
        }
        return;
    }
}
