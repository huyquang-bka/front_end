import streamlit as st
from OCR import *
from Plate import *
from PIL import Image

logo = Image.open("pikachu-logo-619ACB690E-seeklogo.com.png")
st.set_page_config(layout="wide", page_title="AUTOMATIC LICENSE PLATE RECOGNITION BY HUY QUANG", page_icon=logo)
st.title("automatic license plate recognition".upper())
st.sidebar.title("SIDE BAR")
st.sidebar.markdown("-" * 20)
link = '[Go to github repository and give me a star \N{grinning face with smiling eyes} ](https://github.com/huyquang-bka/front_end)'
st.sidebar.markdown(f"{link}", unsafe_allow_html=True)
st.sidebar.markdown("-" * 20)
st.sidebar.markdown("Upload license plate image or Car with license plate here!\n Only for horizontal plate".upper())
file_uploader = st.sidebar.file_uploader("", ["jpg", "png", "jpeg", "gif"])

if file_uploader is not None:
    image = file_uploader.read()
    with open(f"ImageStorage/{file_uploader.name}", "wb+") as f:
        f.write(image)
    img = cv2.imread(f"ImageStorage/{file_uploader.name}")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.sidebar.image(img)
    st.image(img, use_column_width=True)
    try:
        status, crop = plate_detection(img_copy)
        st.image(crop, use_column_width=True)
        st.markdown("-" * 20)
        block_list_check = st.checkbox("Block list detection (Number + Alphaber UpperCase + special characters)")
        if block_list_check:
            vn = st.checkbox("Use Vietnamese License Plate")
            if vn:
                default_list = ["I", "J", "O", "Q", "W", "R"]
                label = "In Vietnamese, On civilian license plates, the letters I, J, O, Q, W, R are not used"
            else:
                default_list = None
                label = ""
            block_list = st.multiselect(label=label,
                                        options=[char for char in (string.ascii_uppercase + string.digits)],
                                        default=default_list)
        else:
            block_list = None

        allow_list = process_allowlist(block_list)
        lp_text = st.button("LP to text")
        if lp_text:
            thre_mor, text = process_image_chracter(crop)
            st.image(thre_mor, use_column_width=True)
            st.title(f"**LP Number: {text}**".upper())
    except:
        st.sidebar.markdown("**CAN'T FIND License Plate**".upper(), unsafe_allow_html=True)
        st.markdown("**CAN'T FIND License Plate**".upper(), unsafe_allow_html=True)
