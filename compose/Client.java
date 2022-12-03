package compose;

import java.util.List;

public class Client {
    public static void main(String[] args) {
        //组织目录结构
        OssDirectory root = new OssDirectory("/root");
        OssDirectory dir1 = new OssDirectory(root,"/s1");
        OssDirectory dir14 = new OssDirectory(root,"/s1/s4");
        OssDirectory dir145 = new OssDirectory(root,"/s1/s4/s5");
        OssDirectory dir2 = new OssDirectory(root,"/s2");
        OssDirectory dir3 = new OssDirectory(root,"/s3");
        //组织文件存放
        OssFile fil1 = new OssFile(dir1, "/f1.txt");
        OssFile fil2 = new OssFile(dir2, "/f2.png");
        OssFile fil3 = new OssFile(dir3, "/f3.gif");
        OssFile fil14 = new OssFile(dir14, "/f14.txt");
        OssFile fil145 = new OssFile(dir145, "/f145.svg");
        //实例化客户端递归打印所有节点路径
        Client client = new Client();
        client.printNodes(root);

    }

    /**
     * 递归打印所有节点路径
     * @param dir
     */
    public void printNodes(OssDirectory dir){
        List<OssNode> children = dir.getChildren();
        for (OssNode node : children) {
            System.out.println(node.getPath());
            if (node instanceof OssDirectory) {
                this.printNodes((OssDirectory) node);
            }
        }
    }
}
