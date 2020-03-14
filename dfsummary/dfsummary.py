from .dfsummary_helpers import return_df_summary, return_heatmap_data

from matplotlib import cm, pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class DfSummary(object):
    """The DfSummary object is a dataframe object with methods to provide
    descriptive statistics and formatted visualisations for EDA."""

    # Initialise class variables for formatting of visualisations
    ncols=3                         # Default number of columns in subplots
    cmap=cm.get_cmap('tab10')       # Default colormap
    fs=9                            # Default font size

    def __init__(self, df):
        """Method for initialising a DfSummary object.

        Args:
        df: Pandas Dataframe.  A dataframe containing continuous and/or discrete data.
        A Pandas Series will be cast as a dataframe.

        Returns:
        DfSummary object
        """

        # If data df is (incorrectly) a Series, cast as a dataframe
        if isinstance(df, pd.Series):
            self.df=df.to_frame()
        else:
            self.df=df

        # Calculate number of subplots rows required for visulisations of numeric data
        self.nrows = int(np.ceil(len(self.df.select_dtypes(include=np.number).columns) \
                                 / (1.0*self.ncols)))
        if self.nrows<=1:
            self.nrows +=1

    def return_summary(self):
        """Display df summary information and descriptive statistics to screen.

        This method also calls the return_df_summary(df) method, which
        returns additional descriptive statitics for the df.  These statistics are
        assigned to the df_summary attribute of the DfSummary object.  Any non-numeric
        columns to be excluded from this summary should be excluded from the
        DfSummary object before instantiation.

        Args:
        None

        Returns:
        None
        """
        print(f'Dataframe shape: {self.df.shape}')
        print(f'Total number of null values in data: {self.df.isnull().sum().sum()}')
        print('\nFirst 5 rows of data:')
        display(self.df.head())
        print('\nLast 5 rows of data:')
        display(self.df.tail())
        print('\n'+'-'*12+'\nDATA SUMMARY\n'+'-'*12)
        # Call function to display and assign additional descriptive statistics for df
        self.df_summary = return_df_summary(self.df)
        display(self.df_summary)

    def return_histograms(self):
        """Create and return a figure with histograms for each numeric column in
        the DfSummary object.

        All NaN values are excluded from histograms.  The number of NaN values
        dropped for each field is displayed in the subplot title for each field.

        Args:
        None

        Returns:
        fig: fig object.  A figure that includes histograms with summary statistics for
        each folumn in the DfSummary object.
        """

        # Initialise figure and axes
        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(15, self.nrows*4))

        # Initialise subplot counter to enable removal of unrequired subplot axes
        counter = 0
        # Assign numeric data for histograms
        df=self.df.select_dtypes(include=np.number)

        # Loop through each subplot cell in axes grid
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = axes[i][j]

                # Generate histogram for each df column, including vertical lines for the
                # mean and 25th, 50th & 75th percentiles.  Note, null values are excluded.
                if counter < len(df.columns):

                    # Drop null values in column before plotting data
                    col_no_null = df.loc[df[df.columns[counter]].notnull(), df.columns[counter]]
                    ax.hist(col_no_null,  color=self.cmap(counter%10), alpha=0.9)
                    ax.axvline(col_no_null.mean(), ls='-', lw=0.9, c='black', label='Mean')
                    ax.axvline(col_no_null.median(), ls='--', lw=0.9, c='black', label='Median')
                    ax.axvline(col_no_null.quantile(0.25), ls='--', lw=0.9, c='dimgray', label='25th percentile')
                    ax.axvline(col_no_null.quantile(0.75), ls='-', lw=0.9, c='dimgray', label='75th percentile')

                    # Configure axis labels and display title
                    ax.set_xlabel('Value', fontsize=self.fs)
                    ax.set_ylabel('Value count', fontsize=self.fs)
                    ax.set_title(f'{df.columns[counter]}: ' \
                                 f'\n({df[df.columns[counter]].isnull().sum()} NaN values dropped)', \
                                 fontsize=self.fs+2)
                    ax.grid(color='lightgray')

                    # Configure legend
                    leg = ax.legend(loc='best', fontsize=self.fs)
                    leg.draw_frame(False)

                # Remove subplot axis for subplot cells without data to plot
                else:
                    ax.set_axis_off()

                # Increment subplot counter
                counter += 1

        # Configure figure layout and assign figure title
        fig.tight_layout(pad=3.0)
        fig.suptitle('Histograms of numeric data fields', fontsize=self.fs+5, y=1.0)

        plt.close(fig)

        return fig

    def return_heatmap(self, method='pearson', drop_criteria=None):
        """Create and return a figure with a heatmap of correlations between numeric columns
        in the DfSummary object.

        This method calls the return_heatmap_data function and assigns the returned correlation
        matrix as the heatmap_data attribute of the DfSummary object.

        Args:
        method: string. Method to be used for calculating correlation between df columns
                ('pearson' (default), 'spearman' or 'kendall')
        drop_criteria: string or list of columns for determining data to drop before calculating
                       correlations.
                      'any_rows': drop any data rows with NaNs
                      'any_cols': drop any data columns with NaNs
                      list: list of columns to be used for determing NaN data to drop. Any rows with
                      NaNs in this list of columns will be dropped from the data.
                      None (default): no rows or columns with NaNs are dropped from the data.
                      If None, pairwise correlations are calculated.

        Returns:
        fig: fig object.  A heatmap of correlation coefficients between each column in df if
        sufficient data remains after dropping NaNs.
        """
        # Assign numeric data for heatmap visulisation
        df=self.df.select_dtypes(include=np.number)
        if df.shape[1]<2 or method.lower() not in ['pearson', 'spearman', 'kendall']:
            print('Invalid data or correlation method provided for heatmap to be generated.')
            return

        # Call function to return data to be included in heatmap based on
        # specified correlation method and null value dropping criteria
        self.heatmap_data = return_heatmap_data(df, method, drop_criteria)

        if self.heatmap_data['corrs'].notnull().sum().sum() == 0:
            print('There is no data left for calcualting pair-wise correlations.')
        else:

            # Initialise figure and axes
            fig, ax = plt.subplots(figsize=(20, 20))

            # Add ticks and axis for colorbar to control its size and position
            cbar_ticks =[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
            cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])

            # Create a mask for the upper triangle of the heatmap
            mask = np.zeros_like(self.heatmap_data['corrs'], dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Generate heatmap
            sns.heatmap(self.heatmap_data['corrs'], mask=mask, vmin=-1, vmax=1, annot=True, annot_kws={"size":self.fs},
                        cmap="RdBu_r", cbar_kws=dict(ticks=cbar_ticks), cbar_ax=cbar_ax, ax=ax)

            # Configure axis labels and assign figure title
            ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=self.fs+4, rotation=90)
            ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=self.fs+4, rotation=0)
            title = (f"Heatmap of {method.capitalize()} correlation between numeric fields with "
                     f"{self.heatmap_data['drop_count']} {self.heatmap_data['title_text']} dropped")
            ax.set_title(title, fontsize=self.fs+7)

            # Adjust font size of colorbar
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=self.fs+4)

            plt.close(fig)

            return fig



    def return_boxplots(self, swarmplot=False):
        """ Create and return a boxplot of data in numeric columns.

        NaN values are excluded from plots.  The number of excluded
        NaN values will be displayed in the subplot title for each field.
        The function has the option to overlay a swarmplot, displaying a maximum of
        2,000 data points.

        Args:
        swarmplot: Boolean (default=False).  Overlay a swarmplot of up to 2,000 data points
                   randomly selected from numeric data columns if swarmplot=True.

        Returns:
        fig: fig object.  Boxplots of data for each numeric data column, with optional swarmplot
        """

        # Initialise figure and axes
        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(14, self.nrows*4))

        # Initialise subplot counter to enable removal of unrequired subplot axes
        counter = 0
        # Assign numeric data for boxplots visulisation
        df=self.df.select_dtypes(include=np.number)

        # Loop through each subplot cell in axes grid
        for i in range(self.nrows):

            for j in range(self.ncols):
                ax = axes[i][j]

                # Generate box and optional swarmplot for each df.column.  Note null values are excluded.
                if counter < len(df.columns):

                    # Drop null values in columnn before plotting data
                    col_no_null = df.loc[df[df.columns[counter]].notnull(), df.columns[counter]]
                    sns.boxplot(data=pd.DataFrame(col_no_null), linewidth=1.2, width=0.75, fliersize=0, \
                                color=self.cmap(counter%10), saturation=0.9, ax=ax)
                    # Generate swarmplot if required
                    if swarmplot and len(col_no_null)<=2000:
                        sns.swarmplot(data=pd.DataFrame(col_no_null), linewidth=0.6, size=0.9, color='darkblue', ax=ax)
                    elif swarmplot and len(col_no_null)>2000:
                        print(f"There are {len(col_no_null)} non-null data points in the '{df.columns[counter]}' field.  "
                              f"2000 random points (only) will be plotted in the swarmplot.")
                        sns.swarmplot(data=pd.DataFrame(col_no_null.sample(2000)), linewidth=0.6, size=0.9, color='darkblue', ax=ax)

                    # Configure axis labels and display title
                    ax.set_xticks([])
                    ax.set_ylabel('Value', fontsize=self.fs)
                    ax.set_title(f'{df.columns[counter]}: \n' \
                                 f'({df[df.columns[counter]].isnull().sum()} NaN values dropped)', fontsize=self.fs+2)
                    ax.grid(color='lightgray')

                # Remove subplot axis for subplot cells without data
                else:
                    ax.set_axis_off()

                # Increment subplot counter
                counter += 1

        # Configure figure layout and assign figure title
        fig.tight_layout(pad=3.0)
        if swarmplot:
            plot_type = 'Swarmplots'
        else:
            plot_type = 'Boxplots'
        fig.suptitle(f'{plot_type} of numeric data fields', fontsize=self.fs+5, y=1.0)

        plt.close(fig)

        return fig
