
def plot_img(state, img, caption="", title="Image"):
    vis = state["vis"]
    img = img / 2 + 0.5  # unnormalize
    vis.images(img.numpy(), padding=1, opts={"title": title, "caption": caption}, win="image-" + title, nrow=4)
