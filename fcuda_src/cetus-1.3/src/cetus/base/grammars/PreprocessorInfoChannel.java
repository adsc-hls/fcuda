package cetus.base.grammars;

import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

@SuppressWarnings({"unchecked", "cast"})
public class PreprocessorInfoChannel
{
    HashMap lineLists = new HashMap(); // indexed by Token number
    int firstValidTokenNumber = 0;
    int maxTokenNumber = 0;

    public void addLineForTokenNumber( Object line, Integer toknum )
    {
        if ( lineLists.containsKey( toknum ) ) {
            List lines = (List) lineLists.get( toknum );
            lines.add(line);
        }
        else {
            List lines = new ArrayList();
            lines.add(line);
            lineLists.put(toknum, lines);
            if ( maxTokenNumber < toknum.intValue() ) {
                maxTokenNumber = toknum.intValue();
            }
        }
    }

    public int getMaxTokenNumber()
    {
        return maxTokenNumber;
    }
        
    public List extractLinesPrecedingTokenNumber( Integer toknum )
    {
        List lines = new ArrayList();
        if (toknum == null) return lines;       
        for (int i = firstValidTokenNumber; i < toknum.intValue(); i++){
            Integer inti = new Integer(i);
            if ( lineLists.containsKey( inti ) ) {
                List tokenLineVector = (List) lineLists.get( inti );
                if ( tokenLineVector != null) {
                    //Enumeration tokenLines = tokenLineVector.elements();
                    //while ( tokenLines.hasMoreElements() ) {
                    //    lines.add( tokenLines.nextElement() );
                    //}
                    lines.addAll(tokenLineVector);
                    lineLists.remove(inti);
                }
            }
        }
        firstValidTokenNumber = toknum.intValue();
        return lines;
    }

    public String toString()
    {
        StringBuffer sb = new StringBuffer("PreprocessorInfoChannel:\n");
        for (int i = 0; i <= maxTokenNumber + 1; i++){
            Integer inti = new Integer(i);
            if ( lineLists.containsKey( inti ) ) {
                List tokenLineVector = (List) lineLists.get( inti );
                if ( tokenLineVector != null) {
                    //Enumeration tokenLines = tokenLineVector.elements();
                    //while ( tokenLines.hasMoreElements() ) {
                    //    sb.append(inti + ":" + tokenLines.nextElement() + '\n');
                    //}
                    int size = tokenLineVector.size();
                    for (int j = 0; j < size; j++) {
                        sb.append(inti);
                        sb.append(":");
                        sb.append(tokenLineVector.get(j));
                        sb.append("\n");
                    }
                }
            }
        }
        return sb.toString();
    }
}



