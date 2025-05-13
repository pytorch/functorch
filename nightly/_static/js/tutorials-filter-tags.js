$(document).ready(function() {
  // Build an array from each tag that's present
  var tagList = [];

  // Check if tutorial containers exist
  if ($(".tutorials-card-container").length > 0) {
    tagList = $(".tutorials-card-container").map(function() {
      if ($(this).data("tags")) {
        return $(this).data("tags").split(",").map(function(item) {
          return item.trim();
        });
      }
      return [];
    }).get();
  } else {
    console.log("No tutorial card containers found");
  }

  // Flatten the array of arrays
  tagList = [].concat.apply([], tagList);

  function unique(value, index, self) {
    return self.indexOf(value) == index && value != "";
  }

  // Only return unique tags
  var tags = tagList.sort().filter(unique);

  console.log("Found tags:", tags);

  // Add filter buttons to the top of the page for each tag
  function createTagMenu() {
    // Check if "All" button already exists
    if ($(".tutorial-filter-menu .filter-btn[data-tag='all']").length === 0) {
      // Add "All" button first
      $(".tutorial-filter-menu").append("<div class='tutorial-filter filter-btn filter selected' data-tag='all'>All</div>");
    }

    if (tags.length > 0) {
      tags.forEach(function(item){
        // Skip adding "all" tag if it's in the tags list
        if (item.toLowerCase() !== "all") {
          $(".tutorial-filter-menu").append(" <div class='tutorial-filter filter-btn filter' data-tag='" + item + "'>" + item + "</div>");
        }
      });
    } else {
      console.log("No tags found to create menu");
    }
  }

  createTagMenu();

  // Show all tutorials by default (up to 8)
  showTutorials(8);

  // Function to show a specific number of tutorials
  function showTutorials(count) {
    $(".tutorials-card-container").hide();
    $(".tutorials-card-container").slice(0, count).show();
  }

  // Add click handler for filter buttons
  $(".filter-btn").on("click", function() {
    var selectedTag = $(this).data("tag");

    // If "All" button is clicked, clear all selections and show everything
    if (selectedTag === "all") {
      $(".filter-btn").removeClass("selected");
      $(this).addClass("selected");
      showTutorials(8); // Show 8 tutorials when "All" is selected
      return;
    }

    // Remove "All" selection when other tags are clicked
    $(".filter-btn[data-tag='all']").removeClass("selected");

    // Toggle selected class
    $(this).toggleClass("selected");

    // Get all selected tags
    var selectedTags = $(".filter-btn.selected").map(function() {
      return $(this).data("tag");
    }).get();

    // Filter the tutorials
    $(".tutorials-card-container").each(function() {
      var cardTags = $(this).data("tags").split(",").map(function(tag) {
        return tag.trim();
      });

      if (selectedTags.length === 0) {
        // If no filters selected, show all
        $(this).show();
      } else {
        // Show if card has ANY of the selected tags
        var hasSelectedTag = cardTags.some(function(tag) {
          return selectedTags.includes(tag);
        });

        if (hasSelectedTag) {
          $(this).show();
        } else {
          $(this).hide();
        }
      }
    });
  });

  // Remove hyphens if they are present in the filter buttons
  $(".tags").each(function(){
    var tags = $(this).text().split(",");
    tags.forEach(function(tag, i) {
      tags[i] = tags[i].replace(/-/, ' ');
    });
    $(this).html(tags.join(", "));
  });

  // Remove hyphens if they are present in the card body
  $(".tutorial-filter").each(function(){
    var tag = $(this).text();
    $(this).html(tag.replace(/-/, ' '));
  });
});
