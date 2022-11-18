public class APP {
    public static void main(String[] args) {
        Employee king = new Employee();
        king.setName("King");
        Car car = new Car();
        car.setNumber("FBW 381");
        king.setCar(car);
        Employee cloneKing = king.deepClone();
        System.out.println();
        System.out.println("King == CloneKing:" + (king == cloneKing));
        System.out.println("King.car == CloneKing.car:" + (king.getCar() == cloneKing.getCar()));
    }
}
