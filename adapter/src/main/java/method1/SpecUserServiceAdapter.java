package method1;

import com.google.gson.Gson;

import java.util.List;
import java.util.Map;

public class SpecUserServiceAdapter implements SpecUserService{
    private UserService userService;

    public SpecUserServiceAdapter(UserService userService) {
        this.userService = userService;
    }

    @Override
    public String findByJId() {
        Map user = userService.findByID();
        String json = new Gson().toJson(user);
        return json;
    }

    @Override
    public String findJUsers() {
        List<Map> user = userService.findUsers();
        String json = new Gson().toJson(user);
        return json;
    }
}
