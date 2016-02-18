package cetus.hir;

import java.util.Collection;
import java.util.regex.Pattern;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.List;

/**
* PragmaAnnotation is used for annotations of pragma type.
*/
public class PragmaAnnotation extends Annotation {

    private static final long serialVersionUID = 3470L;

    /**
    * Constructs an empty pragma annotation.
    */
    public PragmaAnnotation() {
        super();
    }

    /**
    * Constructs a simple pragma with raw string.
    */
    public PragmaAnnotation(String pragma) {
        super();
        put("pragma", pragma);
    }

    /**
    * Returns the name of this pragma annotation.
    */
    public String getName() {
        return (String)get("pragma");
    }

    /**
    * Checks if the specified keys all exist in the key set.
    */
    public boolean containsKeys(Collection<String> keys) {
        for (String key : keys) {
            if (!containsKey(key)) {
                return false;
            }
        }
        return true;
    }

    /**
    * Returns the string representation of this pragma annotation.
    * @return the string.
    */
    public String toString() {
        if (skip_print) {
            return "";
        }
        String ret = "#pragma ";
        String pragma = get("pragma");
        if (pragma != null) {
            ret += pragma + " ";
        }
        if (this.getClass() == PragmaAnnotation.class) { // pure pragma printing
            for (String key : keySet()) {
                if (!key.equals("pragma")) {
                    ret += key + " " + get(key) + " ";
                }
            }
        }
        return ret;
    }

    /**
    * Returns a pragma annotation object after parsing the given text contents.
    * @param note the text part of a pragma.
    * @return the matching pragma annotation.
    */
    public static PragmaAnnotation parse(String note) {
        PragmaAnnotation ret = null;
        ret = PragmaAnnotation.Event.parse(note);
        if (ret == null) {
            ret = PragmaAnnotation.Range.parse(note);
        }
        if (ret == null) {
            ret = new PragmaAnnotation(note);
        }
        return ret;
    }

    /**
    * Pragma annotation type for event timer.
    * Allowed format in regular expression.
    *       #pragma event [^ ]+ (start|stop)
    */
    public static class Event extends PragmaAnnotation {
        
        private static final long serialVersionUID = 3471L;

        public static Event parse(String note) {
            String[] notes = note.trim().split("[ \t]+");
            if (notes.length == 3 &&
                notes[0].equals("event") &&
                (notes[2].equals("start") || notes[2].equals("stop"))) {
                return new Event(notes[1], notes[2]);
            }
            return null;
        }
        
        public Event(String name, String cmd) {
            super("event");
            put("name", name);
            put("command", cmd);
        }

        public String getName() {
            return (String)get("name");
        }

        public String getCommand() {
            return (String)get("command");
        }

        public String toString() {
            if (skip_print) {
                return "";
            }
            return "#pragma event " + getName() + " " + getCommand();
        }

    }

    /**
    * Pragma annotation type for range annotation.
    * Allowed format:
    *       ID: legal identifier at the program point.
    *       E1,E2: legal expressions at the program point.
    *       #pragma range ID=E1,E2( ID=E1,E2)*
    * TODO: needs entry to parser methods; for now, only takes identifiers.
    */
    public static class Range extends PragmaAnnotation {

        private static final long serialVersionUID = 3472L;

        private static final Pattern id_pattern =
                Pattern.compile("[a-zA-Z][a-zA-Z0-9_]*");

        private static final Pattern literal_pattern =
                Pattern.compile("-*[0-9][0-9]*");

        private static final Pattern unary_pattern =
                Pattern.compile("-*[a-zA-Z][a-zA-Z0-9_]*");

        private static final String plus_inf = "+INF";

        private static final String minus_inf = "-INF";

        private Map<Identifier, RangeExpression> map;

        public static Range parse(String note) {
            Range ret = null;
            String[] notes = note.trim().split("[ \t]+");
            if (notes.length >= 2 && notes[0].equals("range")) {
                ret = new Range();
                for (int i = 1; i < notes.length; i++) {
                    String[]range = notes[i].trim().split("[=,]");
                    if (range.length == 3 &&
                        id_pattern.matcher(range[0]).matches()) {
                        // Symbol is searched when running range analysis.
                        Identifier id = SymbolTools.getOrphanID(range[0]);
                        Expression lb = null, ub = null;
                        if (minus_inf.equals(range[1])) {
                            lb = new InfExpression(-1);
                        } else if (id_pattern.matcher(range[1]).matches()) {
                            lb = SymbolTools.getOrphanID(range[1]);
                        } else if (unary_pattern.matcher(range[1]).matches()) {
                            lb = new UnaryExpression(UnaryOperator.MINUS,
                                    SymbolTools.getOrphanID(
                                            range[1].replace("-", "")));
                        } else if (literal_pattern.matcher(range[1]).matches()){
                            lb = new IntegerLiteral(Integer.parseInt(range[1]));
                        }
                        if (plus_inf.equals(range[2])) {
                            ub = new InfExpression(1);
                        } else if (id_pattern.matcher(range[2]).matches()) {
                            ub = SymbolTools.getOrphanID(range[2]);
                        } else if (unary_pattern.matcher(range[2]).matches()) {
                            ub = new UnaryExpression(UnaryOperator.MINUS,
                                    SymbolTools.getOrphanID(
                                            range[2].replace("-", "")));
                        } else if (literal_pattern.matcher(range[2]).matches()){
                            ub = new IntegerLiteral(Integer.parseInt(range[2]));
                        }
                        if (lb != null && ub != null) {
                            ret.map.put(id, new RangeExpression(lb, ub));
                            continue;
                        }
                    }
                    // Failed matches are just skipped.
                    PrintTools.printlnStatus(0,
                            "[WARNING] Skipping dubious range input:",notes[i]);
                }
            }
            return ret;
        }

        public Range() {
            super("range");
            map = new LinkedHashMap<Identifier, RangeExpression>();
        }

        public Map<Identifier, RangeExpression> getMap() {
            return map;
        }

        public String toString() {
            String ret = "#pragma range";
            for (Identifier id : map.keySet()) {
                Expression lb = map.get(id).getLB(), ub = map.get(id).getUB();
                ret += " " + id + "=";
                if (lb instanceof UnaryExpression) {
                    ret += "-" + ((UnaryExpression)lb).getExpression();
                } else {
                    ret += lb;
                }
                ret += ",";
                if (ub instanceof UnaryExpression) {
                    ret += "-" + ((UnaryExpression)ub).getExpression();
                } else {
                    ret += ub;
                }
            }
            return ret;
        }

    }

    /**
    * Converts the pragma annotation to a comment annotation.
    */
    public void toCommentAnnotation() {
        if (ir != null) {       // can't process if attahced ir is missing.
            List<Annotation> notes = this.ir.getAnnotations();
            int index = notes.indexOf(this);
            if (index >= 0) {
                CommentAnnotation comment =
                        new CommentAnnotation(this.toString());
                notes.remove(this);
                comment.ir = ir;
                comment.setOneLiner(true);
                notes.add(index, comment);
            }
        }
    }

}
