import java.util.HashMap;
import java.util.Map;

public class JavaIsTrash {
    public static void main(String[] args) {
        Map<String, Integer> items = new HashMap<String, Integer>(){{
            put("hello", 1);
            put("world", 2);
        }};
        items.put("time", 4);
        int time = (int)items.get("time");
        System.out.println(time);
    }
}
