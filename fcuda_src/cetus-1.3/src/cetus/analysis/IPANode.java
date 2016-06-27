package cetus.analysis;

import cetus.hir.*;

import java.util.*;

/**
 * Class IPANode represents workspace for a procedure, used in interprocedural
 * analysis. It is built upon the DFANode data structure which maintains a
 * generic map for satellite data. IPANode also use the mapped data internally
 * while exposing such data through public methods.
 */
public class IPANode extends DFANode {

    // Exceptions (or qualifiers).
    private static final int MAIN_FUNCTION = 1;
    private static final int VARIABLE_ARG_LIST = 2;
    private static final int WITHIN_CYCLE = 4;

    // The procedure associated with this node.
    private Procedure procedure;

    // The list of formal parameters.
    private List<Symbol> parameters;

    // The list of call sites in the procedure.
    private List<CallSite> call_sites;

    // The list of calling sites that call this procedure.
    private List<CallSite> calling_sites;

    // The map from a set of calling sites to the corresponding IN data.
    // This form is intended for context-sensitive analysis.
    private Map<Set<CallSite>, Domain> in;

    // The OUT data.
    private Domain out;

    // Exception code.
    private int exception;

    // The number of visits to this node.
    private int num_visits;

    /** 
    * Constructs an IPANode object with the given procedure.
    */
    public IPANode(Procedure proc) {
        super();
        procedure = proc;
        call_sites = new LinkedList<CallSite>();
        calling_sites = new LinkedList<CallSite>();
        in = new LinkedHashMap<Set<CallSite>, Domain>();
        out = NullDomain.getNull();
        exception = 0;
        num_visits = 0;
        if (procedure.getParameters().toString().contains("...")) {
            exception |= VARIABLE_ARG_LIST;
        }
        buildParameterList();
        // just for use of base class's method.
        //putData("name", proc.getName());
    }
    
    private void buildParameterList() {
        parameters = new LinkedList<Symbol>();
        for (Object o : procedure.getParameters()) {
            if (o.toString().trim().equals("void")) {
                continue;
            }
            VariableDeclaration vdecln = (VariableDeclaration)o;
            if (vdecln.getNumDeclarators() == 1) {
                parameters.add((Symbol)vdecln.getChildren().get(0));
            } else {
                PrintTools.printlnStatus(0,
                        "[WARNING] multiple symbols in one declaration");
            }
        }
    }

    /**
    * Returns the list of call sites present in the procedure.
    * @return the list of call sites
    */
    public List<CallSite> getCallSites() {
        return call_sites;
    }

    /**
    * Returns the name of the procedure.
    */
    public String getName() {
        return getProcedure().getName().toString();
    }

    /**
    * Returns the procedure associated with the current node.
    * @return a procedure.
    */
    public Procedure getProcedure() {
        return procedure;
    }

    /**
    * Returns the call site object associated with the given function call.
    * @return a call site object.
    */
    public CallSite getCallSite(FunctionCall fc) {
        for (CallSite call_site : getCallSites()) {
            if (call_site.getFunctionCall() == fc) {
                return call_site;
            }
        }
        return null;
    }

    /**
    * Returns the set of IPANode objects that call the current node.
    * @return the set of IPANode.
    */
    public Set<IPANode> getCallers() {
        Set<IPANode> ret = new LinkedHashSet<IPANode>();
        for (DFANode pred : getPreds()) {
            ret.add((IPANode)pred);
        }
        return ret;
    }

    /**
    * Returns the set of CallSite objects that calls the current node.
    * @return the set of calling sites.
    */
    public List<CallSite> getCallingSites() {
        return calling_sites;
    }

    /**
    * Returns the list of formal parameter symbols.
    */
    public List<Symbol> getParameters() {
        return parameters;
    }

    /**
    * Returns the context-sensitive IN data.
    * @param calling_site the calling site of this node.
    * @return the context-sensitive IN data associated with calling_site.
    */
    @SuppressWarnings("unchecked")
    public<T extends Domain> T in(CallSite calling_site) {
        Set<CallSite> calling_set = getCallingSet(calling_site);
        if (calling_set != null) {
            return (T)in.get(calling_set);
        }
        if (isRoot()) {
            return (T)this.in();
        } else {
            return null;
        }
    }

    // Returns the set of calling sites that contains the specified calling
    // site. The returned set is one of the keys used in the IN data.
    private Set<CallSite> getCallingSet(CallSite calling_site) {
        for (Set<CallSite> calling_sites : in.keySet()) {
            if (calling_sites.contains(calling_site)) {
                return calling_sites;
            }
        }
        return null;
    }

    /**
    * Returns the IN data after joining all IN data.
    */
    @SuppressWarnings("unchecked")
    public<T extends Domain> T in() {
        Domain ret = null;
        // Check if context information is available in the node.
        CallSite context = IPAnalysis.getContext(this);
        if (context != null) {
            ret = in(context);
        // Return predefined in data for the main function.
        } else if (isRoot()) {
            ret = this.getData("root-data");
        // Otherwise make conservative decision.
        // This allows main function to get an empty domain which is stored with
        // a null key and an empty domain.
        } else {
            for (Set<CallSite> calling_sites : in.keySet()) {
                Domain curr = in.get(calling_sites);
                if (ret == null) {
                    ret = in.get(calling_sites);
                } else {
                    ret = ret.merge(in.get(calling_sites));
                }
            }
        }
        return (T)ret;
    }

    /**
    * Returns the OUT data.
    */
    @SuppressWarnings("unchecked")
    public<T extends Domain> T out() {
        return (T)out;
    }

    /**
    * Set the IN data with the specified Domain object and the calling site.
    */
    public void in(CallSite calling_site, Domain domain) {
        Domain prev = in(calling_site);
        if (prev == null || !prev.equals(domain)) {
            if (prev != null) { // remove calling_site from the group.
                Set<CallSite> calling_set = getCallingSet(calling_site);
                if (calling_set.size() > 1) {
                    getCallingSet(calling_site).remove(calling_site);
                } else {
                    in.remove(calling_set);
                }
            }
            Set<CallSite> new_calling_set = new LinkedHashSet<CallSite>();
            new_calling_set.add(calling_site);
            in.put(new_calling_set, domain);
        } // else => no update is necessary.
    }

    /**
    * Set the OUT data with the specified Domain object.
    */
    public void out(Domain domain) {
        out = domain;
    }

    /**
    * Removes the IN/OUT data.
    */
    public void clean() {
        in = new LinkedHashMap<Set<CallSite>, Domain>();
        out = NullDomain.getNull();
        for (CallSite call_site : call_sites) {
            call_site.clean();
        }
    }

    /** Marks this node as a root node */
    public void setRoot() {
        exception |= MAIN_FUNCTION;
    }

    /** Marks this node as a cloneable node -- not within any call cycle */
    public void setCloneable(boolean cloneable) {
        if (!cloneable)
            exception |= WITHIN_CYCLE;
    }

    /** Marks this node as one with variable argument list */
    public boolean containsVarArg() {
        return ((exception & VARIABLE_ARG_LIST) != 0);
    }

    /** Checks if this node is a root node */
    public boolean isRoot() {
        return ((exception & MAIN_FUNCTION) != 0);
    }

    /** Checks if this node can be called recursively */
    public boolean isRecursive() {
        return ((exception & WITHIN_CYCLE) != 0);
    }

    /** Increments the number of visits to this node. */
    protected IPANode countVisits() {
        num_visits++;
        return this;
    }

    /** Returns the total number of visits to this node. */
    protected int getVisitCount() {
        return num_visits;
    }

    /** Returns a string dump of this node */
    @Override
    public String toString() {
        StringBuilder str = new StringBuilder(80);
        str.append("[IPANode] ").append(getName());
        str.append(" #CallSites = ").append(call_sites.size()).append("\n");
        str.append("    IN =\n");
        if (isRoot()) {
	    String this_in_str = this.in();
	    str.append("        ").append(this_in_str).append("\n");
        } else {
            for (Set<CallSite> calling_sites : in.keySet()) {
                Domain data_in = in.get(calling_sites);
                for (CallSite calling_site : calling_sites) {
                    str.append("        ").append(calling_site.getID());
                    str.append("  ").append(data_in).append("\n");
                }
            }
        }
	String out_str = out();
        str.append("    OUT =\n").append("        ").append(out_str).append("\n");
        str.append("    Calls =\n");
        for (CallSite site : getCallSites()) {
	    String site_in_str = site.in();
	    String site_out_str = site.out();
            str.append("        ").append(site.getID()).append("\n");
            str.append("            IN = ").append(site_in_str).append("\n");
            str.append("            OUT = ").append(site_out_str).append("\n");
        }
        return str.toString();
    }
}
