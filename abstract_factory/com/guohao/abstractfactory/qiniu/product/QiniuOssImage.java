package abstract_factory.com.guohao.abstractfactory.qiniu.product;

import abstract_factory.com.guohao.abstractfactory.factory.product.OssImage;

public class QiniuOssImage implements OssImage {
    private byte[] bytes;
    public QiniuOssImage(byte[] bytes,String watermark){
        this.bytes = bytes;
        System.out.println("[七牛云]图片已上传至七牛云OSS，URL：http://oss.qiniu.com/xxxxxxx.jpg");
        System.out.println("[七牛云]已生成缩略图，尺寸800X600像素");
        System.out.println("[七牛云]已为图片新增水印，水印文本：" + watermark + ",文本颜色#cccccc");
        System.out.println("[七牛云]已将图片AI增强为1080P高清画质");
    }
    @Override
    public String getThumb() {
        return "http://oss.qiniu.com/xxxxxxx_thumb.jpg";
    }

    @Override
    public String getWatermark() {
        return "http://oss.qiniu.com/xxxxxxx_watermark.jpg";
    }

    @Override
    public String getEnhance() {
        return "http://oss.qiniu.com/xxxxxxx_enhance.jpg";
    }
}
