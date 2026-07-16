"""bsc - Bone-Surface Coordinate cartilage segmentation (Stage 1 / MVP).

Biet lap hoan toan voi phan con lai cua repo. Doc data/model cu READ-ONLY.

Gia thuyet: neo he toa do vao be mat xuong (1 truc theo phap tuyen, 2 truc theo mat)
=> sun tro thanh bai toan 1D doc chieu day. Ky vong cai thien o BIEN va DO DAY noi
sun cuc mong / mat han, KHONG phai o Dice tong the.

Dung tu Colab:
    from google.colab import drive; drive.mount('/content/drive')
    !git clone <repo> /content/repo
    import sys; sys.path.insert(0, "/content/repo")
    from bsc import core, metrics

Xem bsc/README.md cho thu tu cac phase va cac cong chan.
"""

__version__ = "0.1.0"
