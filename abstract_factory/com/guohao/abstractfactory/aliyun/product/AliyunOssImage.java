package abstract_factory.com.guohao.abstractfactory.aliyun.product;

import abstract_factory.com.guohao.abstractfactory.factory.product.OssImage;

public class AliyunOssImage implements OssImage {
    private byte[] bytes;
    public AliyunOssImage(byte[] bytes, String watermark, boolean transparent) {
        this.bytes = bytes;
        System.out.println("[阿里云]图片已上传至阿里云OSS，URL：http://oss.aliyun.com/xxxxxxx.jpg");
                System.out.println("[阿里云]已生成缩略图，尺寸640X480像素");
        System.out.println("[阿里云]已为图片新增水印，水印文本：" + watermark +
                ",文本颜色：#aaaaaa,背景透明：" + transparent);
        System.out.println("[阿里云]已将图片AI增强为4K极清画质");
    }

    @Override
    public String getThumb() {
        return "http://oss.aliyun.com/xxxxxxx_thumb.jpg";
    }

    @Override
    public String getWatermark() {
        return "http://oss.aliyun.com/xxxxxxx_watermark.jpg";
    }

    @Override
    public String getEnhance() {
        return "http://oss.aliyun.com/xxxxxxx_enhance.jpg";
    }
}
