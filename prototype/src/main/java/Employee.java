import com.google.gson.Gson;

public class Employee{
    private String name;
    private Car car;
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Car getCar() {
        return car;
    }

    public void setCar(Car car) {
        this.car = car;
    }

    public Employee deepClone(){
        Gson gson = new Gson();
        String json = gson.toJson(this);
        System.out.println(json);
        Employee cloneObject = gson.fromJson(json, Employee.class);
        return cloneObject;
    }
}
