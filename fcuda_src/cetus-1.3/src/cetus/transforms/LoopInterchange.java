package cetus.transforms;

import cetus.analysis.DDGraph;
import cetus.analysis.DependenceVector;
import cetus.analysis.LoopTools;
import cetus.hir.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * Exchange loops if they are perfect nested loop.
 */
public class LoopInterchange extends TransformPass
{
//    protected Program program;

    public LoopInterchange(Program program)
    {
        super(program);
    }

    public String getPassName()
    {
        return new String("[LoopInterchange]");
    }

    public void start()
    {
        LinkedList<Loop> loops = new LinkedList<Loop>();
        List<Statement> outer_loops = new ArrayList<Statement>();
        List<DependenceVector> depVec = new ArrayList<DependenceVector>();
        DepthFirstIterator iter = new DepthFirstIterator(program);
        List<Expression> expList = new LinkedList<Expression>();
        int i;
        int target_loops = 0;
        int num_single = 0, num_non_perfect = 0, num_contain_func = 0, num_loop_interchange=0;

        while(iter.hasNext()) {
            Object o = iter.next();
            if(o instanceof ForLoop)
                outer_loops.add((Statement)o);
        }

        for(i = outer_loops.size()-1; i >= 0; i--)
        {
            if(!LoopTools.isOutermostLoop((ForLoop)outer_loops.get(i)))
            {
                outer_loops.remove(i);
            }
        }

        System.out.println("# of Outermost Loop: " + outer_loops.size());

        for(i = outer_loops.size()-1; i >= 0; i--)
        {
            iter = new DepthFirstIterator(outer_loops.get(i));
            loops.clear();
            while(iter.hasNext()) {
                Object o = iter.next();
                if(o instanceof ForLoop)
                    loops.add((Loop)o);
            }
            if(loops.size() < 2) {
                num_single++;
            }    else if(!LoopTools.isPerfectNest((ForLoop)loops.get(0))) {
                num_non_perfect++;
            }    else if(LoopTools.containsFunctionCall((ForLoop)loops.get(0))) {
                num_contain_func++;
            } else {
                target_loops++;
                Statement stm = ((ForLoop)loops.get(loops.size()-1)).getBody();
                List<ArrayAccess> arrays = new ArrayList<ArrayAccess>();  // Arrays in loop body
                DepthFirstIterator iter2 = new DepthFirstIterator(stm);

                while(iter2.hasNext())
                {
                    Object child = iter2.next();
                    if(child instanceof ArrayAccess)
                    {
                        arrays.add((ArrayAccess)child);
                    }
                }

                int r = 0,j,until = loops.size();
                int target_index = 0;
                boolean icFlag = true;
                List<Integer> rank;
                int rankSize;

                while(icFlag)
                {
                    Expression exp;
                    expList.clear();
                    for(j = 0; j < until; j++)
                    {
                        exp = LoopTools.getIndexVariable((ForLoop)loops.get(j));
                        if(exp != null)
                            expList.add(exp);
                    }
                    rank = getRank(arrays, expList, target_index);
                    rankSize = rank.size();
                    for(j = 0; j < rankSize; j++) 
                    {
                        r = getRank2(rank, expList, loops);
                        rank.remove(rank.indexOf(r));
                        ///////////////////////////////////////////////////////////////////////////////////
                        // FIXME: requested by SY Lee                                                    //
                        // If the list "rank" consists of r1 and r2 (r1 < r2), and if getRank2() returns //
                        // r1 first, the loop pointed by r2 will be moved upward by 1. Thus r2 in the    //
                        // list "rank" should be changed with r2-1.                                      //
                        ///////////////////////////////////////////////////////////////////////////////////

                        if(expList.size() < until) until = expList.size();
                        for(int k = r+1; k < until; k++)
                        {
                            if(isLegal(loops, r, k))
                            {
                                swapLoop((ForLoop)loops.get(r), (ForLoop)loops.get(k));
                                num_loop_interchange++;
                                Collections.swap(expList, r, k);
                                r = k;

                            } else {
                                break;
                            }
                        }        
                        until = r;
                    }
                    target_index++;
                    if(until == 0) icFlag = false;
                }
            }
        }

        System.out.println("Target loops : " + target_loops);
        System.out.println("Non Perfect loops : " + num_non_perfect);
        System.out.println("Single loops : " + num_single);
        System.out.println("Contain Function : " + num_contain_func);
        System.out.println("Loop Interchanged : " + num_loop_interchange);

        return;
    }

    // Find out which loops could legally interchanged with innermost loop.
    protected List<Integer> rankByMaxInterchange(List<Integer> rank, List<Loop> loops)
    {
        int i, j, legal, max = 0, cur;
        boolean legality;
        List<Integer> result = new LinkedList<Integer>();

        for(i=0; i<rank.size(); i++)
        {
            legal = rank.get(i);
            cur = rank.get(i);
            if(cur+1 == loops.size()) {
                if(legal > max) result.clear();
                result.add(rank.get(i));
                max = legal;
            }

            for(j=cur+1; j<loops.size(); j++)
            {
                legality = isLegal((LinkedList<Loop>)loops, cur, j);
                if(legality) legal = j;
                if(!legality || j == loops.size()-1) {
                    if(legal > max)
                    {
                        result.clear();
                        result.add(rank.get(i));
                        max = legal;
                    } else if (legal == max) {
                        result.add(rank.get(i));
                    }
                    break;
                }
            }
        }
        if(result.size() == 0) return rank;
        return result;
    }

    /* if all upperbound, lowerbound and increment are constnant, we can decide by number of iterations */
    protected List<Integer> rankByNumOfIteration(List<Integer> rank, List<Loop> loops)
    {
        int i, rankSize;
        boolean flag = true;
        Expression lBound, uBound, inc;
        long lb, ub, in, max = 0;
        long count[] = new long[rank.size()];
        List<Integer> result = new LinkedList<Integer>();

        for(i = 0; i < rank.size(); i++)
        {
            if(LoopTools.isUpperBoundConstant(loops.get(rank.get(i))) && LoopTools.isLowerBoundConstant(loops.get(rank.get(i)))
                    && LoopTools.isIncrementConstant(loops.get(rank.get(i))))
            {
                lBound = LoopTools.getLowerBoundExpression(loops.get(rank.get(i)));
                uBound = LoopTools.getUpperBoundExpression(loops.get(rank.get(i)));
                inc = LoopTools.getIncrementExpression(loops.get(rank.get(i)));
                lb = ((IntegerLiteral)lBound).getValue();
                ub = ((IntegerLiteral)uBound).getValue();
                in = ((IntegerLiteral)inc).getValue();
                count[i] = (ub-lb)/in;
            } else {
                flag = false;
                break;
            }
        }

        // check which one has more loop count
        if(flag) {
            for(i = 0; i < rank.size(); i++)
            {
                if(count[i] > max) {
                    result.clear();
                    result.add(rank.get(i));
                    max = count[i];
                } else if (count[i] == max) {
                    result.add(rank.get(i));
                }
            }
        }
        if(result.size() == 0) return rank;
        return result;
    }

    protected int getRank2(List<Integer> rank, List<Expression> expList, List<Loop> loops)
    {
        int i;
        List<Integer> result;

        if(rank.size() == 1) {
            return rank.get(0);
        }

        result = rankByMaxInterchange(rank, loops);
        if(result.size() == 1) return result.get(0);

        result = rankByNumOfIteration(result, loops);
        if(result.size() == 1) return result.get(0);

        return result.get(result.size()-1);
    }

    /* if n = 0, it means right most index in array. */
    protected List<Integer> getRank(List<ArrayAccess> array , List<Expression> expList, int n)
    {
        int i, j, max = 0, cur_exp;
        ArrayList<Integer> result = new ArrayList<Integer>();
        List<Expression> temp = new LinkedList<Expression>();
        Traversable parentTemp;
        Expression lhs, rhs;


        for(i = 0; i < expList.size(); i++)
        {
            Expression e = expList.get(i);
            cur_exp = 0;

            for(j = 0; j < array.size(); j++)
            {
                ArrayAccess f = array.get(j);
                if(f.getNumIndices() >= n) {
                    temp = f.getIndex(f.getNumIndices()-1-n).findExpression(e);
                    if(temp.size() >= 1) {
                        cur_exp+=2;
                        parentTemp = (temp.get(0)).getParent();
                        if(parentTemp instanceof BinaryExpression)
                        {
                            if((((BinaryExpression)parentTemp).getOperator()).toString() == "*")
                            {
                                lhs = ((BinaryExpression)parentTemp).getLHS();
                                rhs = ((BinaryExpression)parentTemp).getRHS();

                                if(lhs.equals((Object)e) || rhs.equals((Object)e)) {
                                    cur_exp--;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if(cur_exp > max) {
                max = cur_exp;
                result.clear();
                result.add(i);
            } else if (cur_exp == max)
                result.add(i);
        }
        return result;
    }

    private List<Expression> getIndex(List<ArrayAccess> array, int n)
    {
        int i;
        List<Expression> result = new LinkedList<Expression>();

        for(i = 0; i < array.size(); i++)
        {
            ArrayAccess f = array.get(i);
            if(f.getNumIndices()-n-1 > 0) result.add(f.getIndex(f.getNumIndices()-n-1));
        }
        return result;

    }

    public void swapLoop(ForLoop loop1, ForLoop loop2) 
    {
        loop1.getInitialStatement().swapWith(loop2.getInitialStatement());
        loop1.getCondition().swapWith(loop2.getCondition());
        loop1.getStep().swapWith(loop2.getStep());

        return;
    }

    /* Check legality of loop interchange between src and target. Both src and target are in nest loops and src is outer than target */
    public boolean isLegal(LinkedList<Loop> nest, int src, int target)
    {
        int i, j, next;
        DDGraph ddg;
        String str;
        ArrayList<DependenceVector> dpv;
        DependenceVector dd;
        ddg = program.getDDGraph();
        dpv = ddg.getDirectionMatrix(nest);

        if(src == target) return true;
        if(src > target) {
            i = src;
            src = target;
            target = i;
        }

        for(i = 0; i < dpv.size(); i++)
        {
            dd = dpv.get(i);
            str = dd.toString();
            for(j = 0; j < str.length(); j++)
            {
                if(j == src) next = target;
                else if(j == target) next = src;
                else next = j;

                if(next < str.length()) {
                    if(str.charAt(next) == '>') return false;
                    if(str.charAt(next) == '<') break;
                }
            }
        }

        return true;
    }

}
