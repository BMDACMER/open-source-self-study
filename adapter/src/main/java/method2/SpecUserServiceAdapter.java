package method2;

import com.google.gson.Gson;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class SpecUserServiceAdapter extends UserServiceImpl implements SpecUserService{
    @Override
    public String findJById() {
        Map user = super.findById();
        String json = new Gson().toJson(user);
        return json;
    }

    @Override
    public String findJUsers() {
        List<Map> users = super.findUsers();
        String json = new Gson().toJson(users);
        return json;
    }
}
