package method2;


public class Client {
    public static void main(String[] args) {
        SpecUserService specUserService = new SpecUserServiceAdapter();
//        System.out.println(specUserService.findJUsers());
        System.out.println(specUserService.findJById());
    }
}
