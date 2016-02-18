package cetus.codegen;

import cetus.exec.Driver;
import cetus.hir.*;

import java.util.EnumSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
* This pass looks for Annotations that provide
* enough information to add OpenMP pragmas and
* then inserts those pragmas.
*/
public class ompGen extends CodeGenPass {

    /**
    * Possible options for handling the existing OpenMP and Cetus-internal
    * pragmas.
    */
    private static enum Option {
        REMOVE_OMP_PRAGMA,      // Avoid printing of the existing OpenMP pragma.
        REMOVE_CETUS_PRAGMA,    // Avoid printing of the existing Cetus pragma.
        COMMENT_OMP_PRAGMA,     // Comment out the existing OpenMP pragam.
        COMMENT_CETUS_PRAGMA    // Comment out the existing Cetus pragma.
    }

    /** Set of options for OpenMP generation */
    private static Set<Option> option;

    private static int omp_for_num, option_value;

    // Reads in the command line option.
    static {
        // Option values:
        // 1. Comment omp
        // 2. Remove omp
        // 3. Remove omp/cetus
        option = EnumSet.noneOf(Option.class);

        try {
            option_value = Integer.parseInt(Driver.getOptionValue("ompGen"));
        } catch(NumberFormatException e) {
            option_value = 1;
        }

        switch (option_value) {
        case 1:
            option.add(Option.COMMENT_OMP_PRAGMA);
            break;
        case 2:
            option.add(Option.REMOVE_OMP_PRAGMA);
            break;
        case 3:
            option.add(Option.REMOVE_CETUS_PRAGMA);
            option.add(Option.REMOVE_OMP_PRAGMA);
            break;
        default:
            option.add(Option.REMOVE_CETUS_PRAGMA);
            option.add(Option.COMMENT_OMP_PRAGMA);
        }

        omp_for_num = 0;
    }

    public ompGen(Program program) {
        super(program);
    }

    public String getPassName() {
        return "[ompGen]";
    }

    public void start() {
        processExistingPragmas();
        DFIterator<Statement> iter =
                new DFIterator<Statement>(program, Statement.class);
        iter.pruneOn(VariableDeclaration.class);
        iter.pruneOn(ExpressionStatement.class);
        while (iter.hasNext()) {
            Statement stmt = iter.next();
            if (stmt instanceof ForLoop) {
                genOmpParallelLoops((ForLoop)stmt);
            } else if (stmt instanceof Statement) {
                genDirectTranslation(stmt);
            }
        }
        // Profitability test is on by default
        String profitable_omp = Driver.getOptionValue("profitable-omp");
        if (profitable_omp == null || !profitable_omp.equals("0")) {
            CodeGenPass.run(new ProfitableOMP(program));
        }
    }

    private void genOmpParallelLoops(ForLoop loop) {
        // currently, we check only omp parallel for construct
        // "cetus for" was added with reduction transformation pass.
        if (!loop.containsAnnotation(CetusAnnotation.class, "parallel") &&
            !loop.containsAnnotation(CetusAnnotation.class, "for")) {
            if (option.contains(Option.REMOVE_CETUS_PRAGMA))
                loop.removeAnnotations(CetusAnnotation.class);
            return;
        }
        // if the loop already contains an OpenMP parallel construct,
        // return
        if (loop.containsAnnotation(OmpAnnotation.class, "for")) {
            if (option.contains(Option.REMOVE_CETUS_PRAGMA))
                loop.removeAnnotations(CetusAnnotation.class);
            return;
        }

        OmpAnnotation omp_annot = new OmpAnnotation();
        List<CetusAnnotation> cetus_annots =
            loop.getAnnotations(CetusAnnotation.class);

        for (CetusAnnotation cetus_annot : cetus_annots) {
            omp_annot.putAll(cetus_annot);
        }

        omp_annot.put("for", "true");
        removeAutomaticPrivateVariables(loop, omp_annot);

        loop.annotateBefore(omp_annot);

        if (option.contains(Option.REMOVE_CETUS_PRAGMA))
            loop.removeAnnotations(CetusAnnotation.class);
    }

    /** Performs a simple one-to-one translation from "cetus" to "omp" */
    private void genDirectTranslation(Statement stmt) {
        List<CetusAnnotation> notes =
            stmt.getAnnotations(CetusAnnotation.class);
        if (!notes.isEmpty()) {
            OmpAnnotation omp_note = new OmpAnnotation();
            for (CetusAnnotation note : notes) {
                omp_note.putAll(note);
                if (option.contains(Option.REMOVE_CETUS_PRAGMA))
                    note.detach();
            }
            removeAutomaticPrivateVariables(stmt, omp_note);
            stmt.annotate(omp_note);
        }
    }

    /** Removes any private variables declared withiin the parallel region */
    @SuppressWarnings("unchecked")
    private void
        removeAutomaticPrivateVariables(Statement stmt, OmpAnnotation note) {
        Set<Symbol> locals = SymbolTools.getLocalSymbols(stmt);
        String[]keys = {"private", "lastprivate"};
        for (String key : keys) {
            Set<Symbol> vars = (Set<Symbol>)note.get(key);
            if (vars != null) {
                Iterator<Symbol> iter = vars.iterator();
                while (iter.hasNext()) {
                    Symbol var = iter.next();
                    if (locals.contains(var)) {
                        iter.remove();
                    }
                }
                if (vars.isEmpty()) {
                    note.remove(key);
                }
            }
        }
    }

    /** Process the already existing omp pragmas. */
    private void processExistingPragmas() {
        DFIterator<Annotatable> iter =
                new DFIterator<Annotatable>(program, Annotatable.class);
        iter.pruneOn(VariableDeclaration.class);
        iter.pruneOn(ExpressionStatement.class);
        while (iter.hasNext()) {
            Annotatable ann = iter.next();
            if (option.contains(Option.REMOVE_OMP_PRAGMA) ||
                option.contains(Option.COMMENT_OMP_PRAGMA)) {
                List<OmpAnnotation> notes =
                    ann.getAnnotations(OmpAnnotation.class);
                for (Annotation note : notes) {
                    if (option.contains(Option.COMMENT_OMP_PRAGMA)) {
                        CommentAnnotation comment =
                            new CommentAnnotation(note.toString());
                        comment.setOneLiner(true);
                        ann.annotate(comment);
                    }
                    note.detach();
                }
            }
        }
    }

}
