import requests
from datetime import datetime

def upload_rsu_data(latitude, longitude, license_plate, flag, deviceNum = "0003",url="http://mileage.cttic-bd.com/zylgate/api/dataapp/rsu/deal"):
    """
    上传RSU数据到指定API。

    :param latitude: 纬度 (str)
    :param longitude: 经度 (str)
    :param license_plate: 车牌号 (str)
    :param flag: 是否信任RSU数据 (str, "1"或"0")
    :param url: API地址 (str)
    :return: 响应内容或错误信息
    """
    params = {
        "deviceNum": deviceNum,
        "longitude": longitude,
        "latitude": latitude,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "licensePlate": license_plate,
        "flag": flag
    }

    try:
        print(f"正在向 {url} 发送数据: {params}")
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("响应数据:", response.json())
            return response.json()
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print("错误信息:", response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return None

# 示例调用
# upload_rsu_data("20.0284", "110.3034", "晋AP79V6", "1")