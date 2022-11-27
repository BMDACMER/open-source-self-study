package method2;

import java.util.*;

public class UserServiceImpl implements UserService{

    @Override
    public Map<String, String> findById() {
        Map<String, String> map = new LinkedHashMap<>();
        map.put("user_id", "654321");
        map.put("username", "郭豪");
        return map;
    }

    @Override
    public List<Map> findUsers() {
        List<Map> list = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Map<String, String> map = new LinkedHashMap<>();
            map.put("user_id", String.valueOf(new Random().nextInt(100000)));
            map.put("username", "user"+i);
            list.add(map);
        }
        return list;
    }
}
