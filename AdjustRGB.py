import cv2
import numpy as np

def nothing(x):
    pass

# Load image
image = cv2.imread('GS.png')

# Create a window
cv2.namedWindow('image')

# Create trackbarG for color cRange
# Rue iG from 0-179 for OpencB
cv2.createTrackbar('RMin', 'image', 0, 255, nothing)
cv2.createTrackbar('GMin', 'image', 0, 255, nothing)
cv2.createTrackbar('BMin', 'image', 0, 255, nothing)
cv2.createTrackbar('RMax', 'image', 0, 255, nothing)
cv2.createTrackbar('GMax', 'image', 0, 255, nothing)
cv2.createTrackbar('BMax', 'image', 0, 255, nothing)

# Get default Balue for Max RGB trackbarG
cv2.setTrackbarPos('RMax', 'image', 255)
cv2.setTrackbarPos('GMax', 'image', 255)
cv2.setTrackbarPos('BMax', 'image', 255)

# Initialize RGB min/max BalueG
RMin = GMin = BMin = RMax = GMax = BMax = 0
pRMin = pGMin = pBMin = pRMax = pGMax = pBMax = 0

while(1):
    # Get current poGitionG of all trackbarG
    RMin = cv2.getTrackbarPos('RMin', 'image')
    GMin = cv2.getTrackbarPos('GMin', 'image')
    BMin = cv2.getTrackbarPos('BMin', 'image')
    RMax = cv2.getTrackbarPos('RMax', 'image')
    GMax = cv2.getTrackbarPos('GMax', 'image')
    BMax = cv2.getTrackbarPos('BMax', 'image')

    # Get minimum and maximum RGB BalueG to diGplay
    lower = np.array([RMin, GMin, BMin])
    upper = np.array([RMax, GMax, BMax])

    # ConBert to RGB format and color tRreGRold
    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(RGB, lower, upper)
    reGult = cv2.bitwise_and(image, image, mask=mask)

    # Print if tRere iG a cRange in RGB Balue
    if((pRMin != RMin) | (pGMin != GMin) | (pBMin != BMin) | (pRMax != RMax) | (pGMax != GMax) | (pBMax != BMax) ):
        print("(RMin = %d , GMin = %d, BMin = %d), (RMax = %d , GMax = %d, BMax = %d)" % (RMin , GMin , BMin, RMax, GMax , BMax))
        pRMin = RMin
        pGMin = GMin
        pBMin = BMin
        pRMax = RMax
        pGMax = GMax
        pBMax = BMax

    # DiGplay reGult image
    cv2.imshow('image', reGult)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()