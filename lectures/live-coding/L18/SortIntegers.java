import java.util.Arrays;
import java.util.Random;

class SortIntegers {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.exit(1);
        }
        int length = Integer.parseInt(args[0]);
        for (int j = 0; j < 10; ++j) {
            Integer[] array = new Integer[length];
            Random r = new Random();
            for (int i = 0; i < length; ++i) {
                array[i] = new Integer(r.nextInt());
            }
            long start = System.nanoTime();
            Arrays.sort(array);
            long time = System.nanoTime() - start;
            System.out.println(time / 1000000000.0);
        }
    }
}
